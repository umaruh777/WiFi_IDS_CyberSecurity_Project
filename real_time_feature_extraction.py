
from scapy.all import *
import pandas as pd
import numpy as np

interface = input("Enter Network Interface: ")
packet_count = 0
data = []

def packet_handler(packet):
    global packet_count
    if packet.haslayer(Dot11):
        pkt_type = packet.type
        pkt_subtype = packet.subtype
        pkt_time = packet.time
        src_mac = packet.addr2
        dst_mac = packet.addr1
        signal_strength = packet.dBm_AntSignal if hasattr(packet, 'dBm_AntSignal') else None

        data.append([pkt_type, pkt_subtype, pkt_time, src_mac, dst_mac, signal_strength])
        packet_count += 1

        if packet_count % 100 == 0:
            df = pd.DataFrame(data, columns=['Type', 'Subtype', 'Time', 'Src_MAC', 'Dst_MAC', 'Signal'])
            df.to_csv("captured_data.csv", index=False)
            print(f"{packet_count} packets processed and saved.")

sniff(iface=interface, prn=packet_handler)
