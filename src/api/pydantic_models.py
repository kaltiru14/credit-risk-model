from pydantic import BaseModel

class PredictRequest(BaseModel):
    CustomerId: str
    Amount: float
    Value: float
    CurrencyCode: str
    CountryCode: int
    ProviderId: str
    ProductCategory: str
    ChannelId: str
    PricingStrategy: int

class PredictResponse(BaseModel):
    CustomerId: str
    risk_probability: float
