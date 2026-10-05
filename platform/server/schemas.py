from datetime import date, time
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

class Payload(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)

class Account(Payload):
    name: str = Field(min_length=2,max_length=100)
    email: str = Field(min_length=5,max_length=254)
    password: str = Field(min_length=10,max_length=128)
    @field_validator('email')
    @classmethod
    def email_valid(cls,v):
        if '@' not in v or '.' not in v.split('@')[-1]: raise ValueError('Enter a valid email address.')
        return v.lower()

class Login(Payload):
    email: str = Field(max_length=254)
    password: str = Field(max_length=128)

class Institution(Payload):
    name: str = Field(min_length=2,max_length=150)
    code: str = Field(min_length=2,max_length=40)
    city: str = Field(min_length=2,max_length=100)
    head: str = Field(min_length=2,max_length=100)
    email: str = Field(min_length=5,max_length=254)
    password: str = Field(min_length=10,max_length=128)
    capacity: int = Field(default=0,ge=0,le=100000)

class Trainer(Account):
    specialty: str = Field(default='',max_length=150)
    institutionId: str | None = None

class Equipment(Payload):
    name: str = Field(min_length=1,max_length=100)
    quantity: int = Field(ge=0,le=10000)
    observable: bool = False

class ClassPayload(Payload):
    name: str = Field(min_length=2,max_length=150)
    course: str = Field(default='',max_length=150)
    room: str = Field(min_length=1,max_length=100)
    date: str
    startTime: str
    endTime: str
    capacity: int = Field(ge=1,le=1000)
    trainerId: str
    institutionId: str | None = None
    expectedEquipment: list[Equipment] = Field(default_factory=list,max_length=100)
    mediaId: str | None = None
    incomingFile: str | None = None
    rtspUrl: str | None = Field(default=None,max_length=2048)
    cameraName: str = Field(default='Classroom camera',max_length=100)
    monitoringEnabled: bool = False
    intervalMinutes: int = Field(default=30,ge=5,le=120)
    tolerance: int = Field(default=2,ge=0,le=20)
    traineeZone: list[float] | None = None
    coverageConfirmed: bool = False
    @model_validator(mode='after')
    def validate_schedule(self):
        date.fromisoformat(self.date)
        start,end=time.fromisoformat(self.startTime),time.fromisoformat(self.endTime)
        if end <= start: raise ValueError('End time must be after start time on the same day.')
        if sum(bool(v) for v in [self.mediaId,self.incomingFile,self.rtspUrl]) != 1: raise ValueError('Choose exactly one video, photo, or camera source.')
        if self.traineeZone is not None:
            z=self.traineeZone
            if len(z)!=4 or not 0<=z[0]<z[2]<=1 or not 0<=z[1]<z[3]<=1: raise ValueError('Trainee zone must be [left,top,right,bottom] normalized to 0–1.')
        return self

class Attendance(Payload):
    submittedCount: int = Field(ge=0,le=1000)
    note: str = Field(default='',max_length=2000)

class Review(Payload):
    decision: Literal['accepted','followup']
    note: str = Field(min_length=3,max_length=2000)

class Settings(Payload):
    intervalMinutes: int = Field(default=30,ge=5,le=120)
    retentionDays: int = Field(default=30,ge=1,le=365)
    uploadRetentionHours: int = Field(default=24,ge=1,le=720)
    tolerance: int = Field(default=2,ge=0,le=20)

class Device(Payload):
    name: str = Field(min_length=2,max_length=100)
    institutionId: str | None = None

class PasswordChange(Payload):
    currentPassword: str = Field(min_length=1,max_length=128)
    password: str = Field(min_length=10,max_length=128)

class PasswordReset(Payload):
    password: str = Field(min_length=10,max_length=128)

class IncomingAssignment(Payload):
    filename: str = Field(min_length=1,max_length=255)
    institutionId: str
