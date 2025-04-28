
from scapy.all import *
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import smtplib
import subprocess

# Load trained model
model = load_model('../model/cnn_model_best.h5')

# Load scaler from training (ensure to save and load a pre-fitted scaler properly in production)
scaler = StandardScaler()
training_data = pd.read_csv('../datasets/preprocessed_features.csv')
scaler.fit(training_data.drop('label', axis=1))

# Email alert function
def send_alert_email():
    sender_email = "youremail@example.com"
    receiver_email = "admin@example.com"
    password = "yourpassword"

    message = "Subject: Deauthentication Attack Alert\n\nA Deauthentication attack has been detected on your network."

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)

# Mitigation by blocking MAC address (example using Linux firewall)
def mitigate_attack(mac_address):
    subprocess.call(['sudo', 'iptables', '-A', 'INPUT', '-m', 'mac', '--mac-source', mac_address, '-j', 'DROP'])
    print(f"Blocked MAC address: {mac_address}")

interface = input("Enter Network Interface: ")

def packet_handler(packet):
    if packet.haslayer(Dot11):
        pkt_features = np.array([[packet.type, packet.subtype]])
        pkt_features_scaled = scaler.transform(pkt_features).reshape(1, -1, 1)

        prediction = model.predict(pkt_features_scaled)
        label = np.argmax(prediction, axis=1)

        if label == 1:  # assuming '1' represents malicious
            print("[ALERT] Deauthentication Attack Detected!")
            attacker_mac = packet.addr2
            send_alert_email()
            mitigate_attack(attacker_mac)

sniff(iface=interface, prn=packet_handler)
