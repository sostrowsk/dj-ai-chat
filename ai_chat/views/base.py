# ai_chat/views/base.py
from typing import Any, Dict

from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import TemplateView
from two_factor.views.mixins import OTPRequiredMixin


class SecureView(LoginRequiredMixin, OTPRequiredMixin, TemplateView):
    login_url = "/login/"
    redirect_field_name = "next"

    def get_context_data(self, **kwargs: Any) -> Dict[str, Any]:
        context = super().get_context_data(**kwargs)
        context["page_title"] = self.page_title
        return context
