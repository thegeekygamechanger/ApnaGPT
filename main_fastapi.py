from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from authenticator import Authenticator

app = FastAPI()

class User(BaseModel):
    username: str
    password: str
    phone: str = None

class Login(BaseModel):
    username: str
    password: str

class ResetPassword(BaseModel):
    phone: str
    new_password: str

@app.post("/login")
def login(user: Login):
    if Authenticator.authenticate_user(user.username, user.password):
        return {"message": "Login successful"}
    raise HTTPException(status_code=401, detail="Invalid username or password")

@app.post("/signup")
def signup(user: User):
    if len(user.phone) == 10 and user.phone.isdigit():
        if Authenticator.register_user(user.username, user.password, user.phone):
            return {"message": "Signup successful"}
        raise HTTPException(status_code=400, detail="User already exists")
    raise HTTPException(status_code=400, detail="Invalid phone number. Please enter a 10-digit phone number.")

@app.post("/forgot-password")
def forgot_password(phone: str):
    username = Authenticator.get_username_by_phone(phone)
    if username:
        return {"username": username}
    raise HTTPException(status_code=404, detail="Phone number not found")

@app.post("/reset-password")
def reset_password(reset: ResetPassword):
    if Authenticator.update_password(reset.phone, reset.new_password):
        return {"message": "Password reset successfully"}
    raise HTTPException(status_code=400, detail="Failed to reset password")
