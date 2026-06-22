"""
DeepSeek API Server - Token Manager (multi-account round-robin)
"""

import threading
from deepseek_client import login
from config import ACCOUNTS

_account_lock = threading.Lock()
_current_account_index = 0


def prelogin_all_accounts():
    """Login all accounts in background at startup to avoid first-request timeout."""
    threading.Thread(target=_token_refresh_loop, daemon=True).start()
    def _login_all():
        for i, acc in enumerate(ACCOUNTS):
            if acc.get("token"):
                continue
            try:
                print(f"[auth] Pre-login tai khoan #{i+1}: {acc.get('email')}")
                token = login(email=acc.get("email"), password=acc.get("password"))
                acc["token"] = token
                print(f"[auth] Pre-login OK #{i+1}: {token[:20]}...")
            except Exception as e:
                print(f"[auth] Pre-login loi #{i+1} ({acc.get('email')}): {e}")
    threading.Thread(target=_login_all, daemon=True).start()


def _token_refresh_loop():
    """Background thread: refresh all tokens every 10 minutes."""
    import time as _time
    while True:
        _time.sleep(600)
        print("[auth] Refreshing all tokens...")
        for i, acc in enumerate(ACCOUNTS):
            try:
                token = login(email=acc.get("email"), password=acc.get("password"))
                with _account_lock:
                    acc["token"] = token
                print(f"[auth] Refresh OK #{i+1}: {token[:20]}...")
            except Exception as e:
                print(f"[auth] Refresh fail #{i+1} ({acc.get('email')}): {e}")
        print("[auth] Token refresh complete")


def get_active_token(force_refresh: bool = False) -> str:
    global _current_account_index
    with _account_lock:
        if not ACCOUNTS:
            raise RuntimeError("Không có tài khoản DeepSeek nào được cấu hình!")

        for _ in range(len(ACCOUNTS)):
            acc = ACCOUNTS[_current_account_index]
            if force_refresh or not acc.get("token"):
                try:
                    print(f"[auth] Login tai khoan #{_current_account_index + 1}: {acc.get('email')}")
                    token = login(
                        email=acc.get("email"),
                        password=acc.get("password")
                    )
                    acc["token"] = token
                    print(f"[auth] Login OK #{_current_account_index + 1}: {token[:20]}...")
                except Exception as e:
                    print(f"[auth] Login fail #{_current_account_index + 1} ({acc.get('email')}): {e}")
                    _current_account_index = (_current_account_index + 1) % len(ACCOUNTS)
                    continue

            token = acc["token"]
            _current_account_index = (_current_account_index + 1) % len(ACCOUNTS)
            return token

        raise RuntimeError("Tat ca tai khoan DeepSeek deu login that bai!")


def invalidate_token(token: str = None):
    with _account_lock:
        if token:
            for acc in ACCOUNTS:
                if acc.get("token") == token:
                    print(f"[auth] Invalidate token của tài khoản: {acc.get('email')}")
                    acc["token"] = None
                    break
        else:
            for acc in ACCOUNTS:
                acc["token"] = None
