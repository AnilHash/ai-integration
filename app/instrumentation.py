from langfuse import get_client


def verify_langfuse_connection() -> bool:
    client = get_client()
    ok = client.auth_check()

    if ok:
        print("Langfuse connected - traces will appear in cloud.langfuse.com")
    else:
        print("Langfuse auth check failed - check LANGFUSE_PUBLIC_KEY, ")
        print("LANGFUSE_SECRET_KEY, and LANGFUSE_HOST in .env")
        print("Requests will still be served; traces will be lost")

    return ok
