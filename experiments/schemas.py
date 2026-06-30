from evergreen.catalog.schema import Field, Schema

DIALOG_ID = Field(
    "dialog_id", str, "The unique identifier of the customer support dialog."
)
COMPANY_ID = Field("company_id", str, "The unique identifier of the airline company.")
DIALOG = Field("dialog", str, "The customer support dialog.")

REVIEW_ID = Field("review_id", str, "The unique identifier of the customer review.")
BUSINESS_ID = Field(
    "business_id", str, "The unique identifier of the reviewed restaurant."
)
TEXT = Field("text", str, "The customer review text.")

DIALOG_SCHEMA = Schema((DIALOG_ID, DIALOG), key=("dialog_id",))

DIALOG_WITH_COMPANY_SCHEMA = Schema((DIALOG_ID, COMPANY_ID, DIALOG), key=("dialog_id",))

REVIEW_SCHEMA = Schema((REVIEW_ID, TEXT), key=("review_id",))

REVIEW_WITH_BUSINESS_SCHEMA = Schema((REVIEW_ID, BUSINESS_ID, TEXT), key=("review_id",))
