import os
import json
import hashlib

USER_DB = "users.json"

class Authenticator:
    @staticmethod
    def load_users():
        if os.path.exists(USER_DB):
            with open(USER_DB, "r") as f:
                users = json.load(f)
            for user in users.values():
                if "phone" not in user:
                    user["phone"] = ""
            return users
        return {}

    @staticmethod
    def save_users(users):
        with open(USER_DB, "w") as f:
            json.dump(users, f, indent=4)

    @staticmethod
    def authenticate_user(username, password):
        users = Authenticator.load_users()
        if username in users:
            hashed_input_password = hashlib.sha256(password.encode()).hexdigest()
            return hashed_input_password == users[username]["password"]
        return False

    @staticmethod
    def register_user(username, password, phone):
        users = Authenticator.load_users()
        if username in users:
            return False
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        users[username] = {"password": hashed_password, "phone": phone}
        Authenticator.save_users(users)
        return True

    @staticmethod
    def update_password(username, new_password):
        users = Authenticator.load_users()
        if username in users:
            hashed_password = hashlib.sha256(new_password.encode()).hexdigest()
            users[username]["password"] = hashed_password
            Authenticator.save_users(users)
            return True
        return False

    @staticmethod
    def get_username_by_phone(phone):
        users = Authenticator.load_users()
        for username, details in users.items():
            if details.get("phone") == phone:
                return username
        return None
