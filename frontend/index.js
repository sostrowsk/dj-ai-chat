// src/js/chat/index.js
import { marked } from 'marked';

class WebSocketClient {
    // Sources beyond this count start collapsed ("show all" toggle).
    static VISIBLE_SOURCES = 5;

    constructor(options = {}) {
        this.options = { reconnectAttempts: 3, reconnectDelay: 2000, debug: false, ...options };
        this.state = {
            connectionAttempts: 0,
            isConnecting: false,
            socket: null,
            messageQueue: [],
            currentMessage: '',
            isWaitingForResponse: false,
            sourcesAdded: false,
        };
        this.elements = {
            messages: document.getElementById('chatMessages'),
            input: document.getElementById('messageInput'),
            sendButton: document.getElementById('sendButton'),
            clearButton: document.getElementById('clearHistory'),
        };
        // Translated labels come from data-* attributes set via {% translate %}
        // in the template — JS itself has no i18n.
        const dataset = this.elements.messages?.dataset ?? {};
        this.labels = {
            sources: dataset.labelSources || 'Sources',
            score: dataset.labelScore || 'Score',
            page: dataset.labelPage || 'Page',
            showAllSources: dataset.labelShowAllSources || 'Show all sources',
            showFewerSources: dataset.labelShowFewerSources || 'Show fewer sources',
        };
        this.initializeConnection();
    }

    initializeConnection() {
        this.setupEventListeners();
        this.connect();
    }

    connect() {
        if (this.state.isConnecting) return;
        this.state.isConnecting = true;
        const wsUrl = `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws/chat/`;

        try {
            this.state.socket = new WebSocket(wsUrl);
            this.setupSocketHandlers();
            this.handleInitialRouteData();
        } catch {
            this.handleConnectionError();
        }
    }

    setupSocketHandlers() {
        const socket = this.state.socket;
        socket.onopen = () => this.handleSuccessfulConnection();
        socket.onmessage = event => this.handleIncomingMessage(event);
        socket.onclose = event => this.handleConnectionClose(event);
        socket.onerror = () => this.showError('Connection error occurred');
    }

    handleInitialRouteData() {
        const generalChatMatch = location.pathname.match(/\/[a-z]{2}\/ai-chat\//);
        const projectMatch = location.pathname.match(/\/[a-z]{2}\/ai-chat\/project\/(\d+)\//);
        const documentMatch = location.pathname.match(/\/[a-z]{2}\/ai-chat\/document\/(\d+)\//);
        if (documentMatch) {
            this.state.messageQueue.push({
                type: 'document',
                document_id: parseInt(documentMatch[1]),
            });
        } else if (projectMatch) {
            this.state.messageQueue.push({
                type: 'project',
                project_id: parseInt(projectMatch[1]),
            });
        } else if (generalChatMatch) {
            this.state.messageQueue.push({
                type: 'general',
            });
        }
    }

    handleSuccessfulConnection() {
        this.state.isConnecting = false;
        this.state.connectionAttempts = 0;
        this.processMessageQueue();
        this.updateConnectionStatus(true);
    }

    handleIncomingMessage(event) {
        try {
            const data = JSON.parse(event.data);
            if (data.error) {
                this.showError(data.error);
                this.removeSpinner();
                return;
            }
            switch (data.type) {
                case 'message':
                    if (data.message === '[EOS]') {
                        this.finishAiMessage();
                        this.state.sourcesAdded = false;
                        if (data.sources?.length && !this.state.sourcesAdded) {
                            this.updateAiSources(data.sources);
                            this.state.sourcesAdded = true;
                        }
                    } else if (data.message) {
                        this.updateAiMessage(data.message);
                        if (data.sources?.length && !this.state.sourcesAdded) {
                            this.updateAiSources(data.sources);
                            this.state.sourcesAdded = true;
                        }
                    }
                    break;
                case 'history':
                    this.renderHistory(data.messages || []);
                    break;
                case 'history_cleared':
                    this.clearHistoryUI();
                    break;
                default:
                    if (data.message === '[EOS]') {
                        this.finishAiMessage();
                        this.state.sourcesAdded = false;
                    } else if (data.message) {
                        this.updateAiMessage(data.message);
                        if (data.sources?.length && !this.state.sourcesAdded) {
                            this.updateAiSources(data.sources);
                            this.state.sourcesAdded = true;
                        }
                    }
            }
        } catch {
            this.showError('Invalid message format');
            this.removeSpinner();
        }
    }

    handleConnectionClose(event) {
        this.state.isConnecting = false;
        this.updateConnectionStatus(false);
        this.state.isWaitingForResponse && this.removeSpinner();
        if ([1006, 1012].includes(event.code)) {
            this.handleConnectionError();
        }
    }

    setupEventListeners() {
        this.elements.sendButton?.addEventListener('click', this.sendMessage.bind(this));
        this.elements.input?.addEventListener('keypress', e => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        this.elements.clearButton?.addEventListener('click', this.clearHistory.bind(this));
    }

    createElement(tag, className, text = '') {
        const element = document.createElement(tag);
        element.className = className;
        text && (element.textContent = text);
        return element;
    }

    createMessageContainer(type) {
        const messageDiv = this.createElement('div', `message ${type}`);
        const contentDiv = this.createElement('div', 'message-content');
        const headerDiv = this.createElement('div', 'message-header', type === 'sent' ? 'You' : 'AI');
        const bodyDiv = this.createElement('div', 'message-body');

        contentDiv.append(headerDiv, bodyDiv);
        messageDiv.appendChild(contentDiv);
        this.elements.messages?.appendChild(messageDiv);
        this.scrollToBottom();
        return bodyDiv;
    }

    updateAiMessage(message) {
        if (!this.state.isWaitingForResponse) {
            this.state.isWaitingForResponse = true;
            this.state.currentMessage = '';
            this.createMessageContainer('received');
        }
        this.state.currentMessage += message;
        this.updateCurrentMessage();
    }

    renderHistory(messages) {
        // Replayed history replaces whatever is rendered (fresh connect or
        // reconnect — avoids duplicated turns).
        if (!this.elements.messages) return;
        this.elements.messages.innerHTML = '';
        this.state.currentMessage = '';
        this.state.isWaitingForResponse = false;
        this.state.sourcesAdded = false;
        messages.forEach(entry => {
            const bodyDiv = this.createMessageContainer(entry.role === 'user' ? 'sent' : 'received');
            bodyDiv.innerHTML = marked.parse(entry.content || '');
            if (entry.role !== 'user' && entry.sources?.length) {
                this.updateAiSources(entry.sources);
            }
        });
        this.scrollToBottom();
    }

    updateAiSources(sources) {
        const messageContainer = this.elements.messages?.querySelector('.message.received:last-child');
        if (!messageContainer) return;
        const visibleCount = WebSocketClient.VISIBLE_SOURCES;
        const sourcesContainer = this.createElement('div', 'sources-container');
        const sourcesHeader = this.createElement('div', 'sources-header', this.labels.sources);
        const sourcesList = this.createElement('div', 'sources-list');
        sources.forEach((source, index) => {
            const extraClass = index >= visibleCount ? ' source-item-extra d-none' : '';
            const sourceItem = this.createElement('div', `source-item${extraClass}`);
            const sourceContent = this.createElement('div', 'source-content');
            const documentTitle = source.name || 'Document';
            const documentId = source.id || 'Unknown';
            const sourceTitle = this.createElement('div', 'source-title');
            const titleLink = this.createElement('a', 'source-link');
            titleLink.href = `/data-room/redirect/${documentId}/`;
            titleLink.target = '_blank';
            titleLink.rel = 'noopener noreferrer';
            titleLink.textContent = `${documentTitle} (ID: ${documentId})`;
            sourceTitle.appendChild(titleLink);
            sourceContent.appendChild(sourceTitle);
            const metaParts = [];
            if (typeof source.score === 'number') {
                metaParts.push(`${this.labels.score}: ${source.score}`);
            }
            if (source.page_number !== null && source.page_number !== undefined) {
                metaParts.push(`${this.labels.page} ${source.page_number}`);
            }
            if (metaParts.length) {
                sourceContent.appendChild(this.createElement('div', 'source-meta', metaParts.join(' · ')));
            }
            sourceItem.appendChild(sourceContent);
            sourcesList.appendChild(sourceItem);
        });
        sourcesContainer.append(sourcesHeader, sourcesList);
        if (sources.length > visibleCount) {
            sourcesContainer.appendChild(this.createSourcesToggle(sourcesList, sources.length));
        }
        messageContainer.appendChild(sourcesContainer);
        this.scrollToBottom();
    }

    createSourcesToggle(sourcesList, totalCount) {
        const toggle = this.createElement(
            'button',
            'btn btn-link btn-sm sources-toggle',
            `${this.labels.showAllSources} (${totalCount})`
        );
        toggle.type = 'button';
        toggle.addEventListener('click', () => {
            const expanded = toggle.classList.toggle('expanded');
            sourcesList
                .querySelectorAll('.source-item-extra')
                .forEach(item => item.classList.toggle('d-none', !expanded));
            toggle.textContent = expanded
                ? this.labels.showFewerSources
                : `${this.labels.showAllSources} (${totalCount})`;
        });
        return toggle;
    }

    updateCurrentMessage() {
        const messageBody = this.elements.messages?.querySelector('.message.received:last-child .message-body');
        if (messageBody) {
            messageBody.innerHTML = marked.parse(this.state.currentMessage);
            this.scrollToBottom();
        }
    }

    finishAiMessage() {
        if (this.state.isWaitingForResponse) {
            this.state.isWaitingForResponse = false;
            this.removeSpinner();
            this.state.currentMessage = '';
            this.enableInput();
        }
    }

    sendMessage() {
        const input = this.elements.input;
        if (!input?.value.trim() || this.state.isWaitingForResponse) return;
        const message = input.value.trim();
        const messageBody = this.createMessageContainer('sent');
        messageBody.innerHTML = marked.parse(message);
        this.elements.messages?.appendChild(this.createElement('div', 'spinner'));
        this.scrollToBottom();
        if (this.state.socket?.readyState === WebSocket.OPEN) {
            this.state.socket.send(JSON.stringify({ type: 'message', message }));
        } else {
            this.state.messageQueue.push({ type: 'message', message });
            !this.state.isConnecting && this.connect();
        }
        input.value = '';
        this.disableInput();
    }

    processMessageQueue() {
        while (this.state.messageQueue.length) {
            this.state.socket.send(JSON.stringify(this.state.messageQueue.shift()));
        }
    }

    updateConnectionStatus(connected) {
        const { sendButton, input } = this.elements;
        if (sendButton) sendButton.disabled = !connected;
        if (input) {
            input.disabled = !connected;
            input.placeholder = connected ? 'Type your message...' : 'Connecting...';
        }
        !connected && this.showError('Disconnected from chat server');
    }

    clearHistory() {
        try {
            this.elements.messages && (this.elements.messages.innerHTML = '');
            this.state.socket?.readyState === WebSocket.OPEN &&
                this.state.socket.send(JSON.stringify({ type: 'clear' }));
            this.enableInput();
            this.elements.input && (this.elements.input.value = '');
            this.state.isWaitingForResponse = false;
            this.state.sourcesAdded = false;
        } catch {
            this.showError('Error clearing history');
        }
    }

    clearHistoryUI() {
        if (this.elements.messages) {
            this.elements.messages.innerHTML = '';
            this.state.currentMessage = '';
            this.state.isWaitingForResponse = false;
            this.state.sourcesAdded = false;
            this.removeSpinner();
            this.enableInput();
        }
    }

    handleConnectionError() {
        if (this.state.connectionAttempts < this.options.reconnectAttempts) {
            this.state.connectionAttempts++;
            setTimeout(() => this.connect(), this.options.reconnectDelay);
        } else {
            this.showError('Failed to establish connection');
        }
    }

    showError(message) {
        const errorElement = this.createElement('div', 'alert alert-danger', message);
        this.elements.messages?.appendChild(errorElement);
        setTimeout(() => errorElement.remove(), 5000);
    }

    enableInput() {
        const { input, sendButton } = this.elements;
        if (input) {
            input.disabled = false;
            input.focus();
        }
        if (sendButton) sendButton.disabled = false;
    }

    disableInput() {
        const { input, sendButton } = this.elements;
        if (input) input.disabled = true;
        if (sendButton) sendButton.disabled = true;
    }

    removeSpinner() {
        this.elements.messages?.querySelector('.spinner')?.remove();
    }

    scrollToBottom() {
        const messages = this.elements.messages;
        messages && (messages.scrollTop = messages.scrollHeight);
    }

    disconnect() {
        this.state.socket?.close(1000);
        this.state.isConnecting = false;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.chatClient = new WebSocketClient({ debug: true });
});
