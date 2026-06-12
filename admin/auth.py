import secrets


def admin_enabled(admin_password: str) -> bool:
    return bool(admin_password)


def check_admin_credentials(admin_password: str, username: str, password: str) -> bool:
    if not admin_enabled(admin_password):
        return False
    return secrets.compare_digest(username or "", "admin") and secrets.compare_digest(password or "", admin_password)
