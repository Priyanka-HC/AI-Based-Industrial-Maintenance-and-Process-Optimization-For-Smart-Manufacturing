"""
Run this ONCE to create dashboard/auth_config.yaml with:
  - a seeded admin account (change its password after first login)
  - a list of pre-authorized employee emails allowed to self-register

Usage (from the dashboard folder):
    python generate_config.py
"""
import yaml
import streamlit_authenticator as stauth

# ---- EDIT THESE ----
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "ChangeMe123!"   # change after first login
ADMIN_EMAIL = "admin@yourcompany.com"

# Only people with these emails will be allowed to sign themselves up.
# Add your teammates' real emails here.
PRE_AUTHORIZED_EMPLOYEE_EMAILS = [
    "pavanithamma@gmail.com",
]
# ---------------------

credentials = {
    "usernames": {
        ADMIN_USERNAME: {
            "email": ADMIN_EMAIL,
            "first_name": "Admin",
            "last_name": "User",
            "password": ADMIN_PASSWORD,  # plain text here — Hasher hashes it below
        }
    }
}

# Hash every password in the credentials dict in place
stauth.Hasher.hash_passwords(credentials)

config = {
    "credentials": credentials,
    "cookie": {
        "name": "pm_dashboard_auth",
        "key": "change_this_to_a_random_secret_string",  # change this too
        "expiry_days": 7,
    },
    "preauthorized": {
        "emails": PRE_AUTHORIZED_EMPLOYEE_EMAILS,
    },
}

with open("auth_config.yaml", "w") as f:
    yaml.dump(config, f, default_flow_style=False)

print("Created auth_config.yaml")
print(f"Admin login -> username: {ADMIN_USERNAME}  password: {ADMIN_PASSWORD}")
print("Change the password after first login, and edit the pre-authorized emails list.")