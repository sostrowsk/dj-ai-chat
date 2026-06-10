class ErrorCodes:

    # Initialization Errors (4000-4099)
    VECTOR_STORE_FAILED = 4001
    CONNECT_FAILED = 4002
    NOT_INITIALIZED = 4004

    # Message Processing Errors (4100-4199)
    INVALID_MESSAGE = 4101
    INVALID_JSON = 4102
    PROCESSING_FAILED = 4103

    # Authentication Errors (4200-4299)
    USER_NOT_FOUND = 4201
    OTP_REQUIRED = 4202

    # Resource Errors (4300-4399)
    PROJECT_NOT_FOUND = 4301
    DOCUMENT_NOT_FOUND = 4302

    # Runtime Errors (4400-4499)
    SETUP_FAILED = 4401
