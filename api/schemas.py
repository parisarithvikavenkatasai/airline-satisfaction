"""Pydantic request/response models. JSON keys use the original dataset column names."""
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class PassengerFeatures(BaseModel):
    model_config = ConfigDict(populate_by_name=True, json_schema_extra={"example": {
        "Gender": "Female", "Customer Type": "Loyal Customer", "Age": 41, "Type of Travel": "Business travel",
        "Class": "Business", "Flight Distance": 1850, "Inflight wifi service": 4,
        "Departure/Arrival time convenient": 4, "Ease of Online booking": 4, "Gate location": 3,
        "Food and drink": 4, "Online boarding": 5, "Seat comfort": 5, "Inflight entertainment": 5,
        "On-board service": 4, "Leg room service": 4, "Baggage handling": 4, "Checkin service": 4,
        "Inflight service": 4, "Cleanliness": 5, "Departure Delay in Minutes": 5,
        "Arrival Delay in Minutes": 3}})

    gender: Literal["Male", "Female"] = Field(alias="Gender")
    customer_type: Literal["Loyal Customer", "disloyal Customer"] = Field(alias="Customer Type")
    age: int = Field(alias="Age", ge=5, le=100)
    type_of_travel: Literal["Business travel", "Personal Travel"] = Field(alias="Type of Travel")
    travel_class: Literal["Business", "Eco", "Eco Plus"] = Field(alias="Class")
    flight_distance: int = Field(alias="Flight Distance", ge=0)
    inflight_wifi_service: int = Field(alias="Inflight wifi service", ge=0, le=5)
    departure_arrival_time_convenient: int = Field(alias="Departure/Arrival time convenient", ge=0, le=5)
    ease_of_online_booking: int = Field(alias="Ease of Online booking", ge=0, le=5)
    gate_location: int = Field(alias="Gate location", ge=0, le=5)
    food_and_drink: int = Field(alias="Food and drink", ge=0, le=5)
    online_boarding: int = Field(alias="Online boarding", ge=0, le=5)
    seat_comfort: int = Field(alias="Seat comfort", ge=0, le=5)
    inflight_entertainment: int = Field(alias="Inflight entertainment", ge=0, le=5)
    on_board_service: int = Field(alias="On-board service", ge=0, le=5)
    leg_room_service: int = Field(alias="Leg room service", ge=0, le=5)
    baggage_handling: int = Field(alias="Baggage handling", ge=0, le=5)
    checkin_service: int = Field(alias="Checkin service", ge=0, le=5)
    inflight_service: int = Field(alias="Inflight service", ge=0, le=5)
    cleanliness: int = Field(alias="Cleanliness", ge=0, le=5)
    departure_delay: float = Field(alias="Departure Delay in Minutes", ge=0)
    arrival_delay: Optional[float] = Field(default=None, alias="Arrival Delay in Minutes", ge=0)


class PredictionResponse(BaseModel):
    prediction: str
    probability_satisfied: float
    model_version: int


class BatchRequest(BaseModel):
    passengers: List[PassengerFeatures] = Field(min_length=1)


class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[int] = None
