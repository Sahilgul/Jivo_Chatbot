import os
import secrets
from fastapi import HTTPException, status, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()


async def verify_username(request: Request) -> HTTPBasicCredentials:
    credentials = await security(request)
    correct_username = secrets.compare_digest(
        credentials.username, os.getenv("USER", "admin")
    )
    correct_password = secrets.compare_digest(
        credentials.password, os.getenv("PASSWORD", "1q2w3E*")
    )
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username
