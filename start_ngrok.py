import time
import ngrok

AUTH_TOKEN = "3It7Q8HBf269U6ESFjr6pkKYig7_41gEnCX1qpeEytpJa6Cd6"

def main():
    ngrok.set_auth_token(AUTH_TOKEN)
    listener = ngrok.forward(8501)
    url = listener.url()
    print(f"PUBLIC_URL={url}", flush=True)
    with open("ngrok_url.txt", "w") as f:
        f.write(url)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
