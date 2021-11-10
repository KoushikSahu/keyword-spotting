from pathlib import Path
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = {
            'backward': 1664,
            'bed': 2014,
            'bird': 2064,
            'cat': 2031,
            'dog': 2128,
            'down': 3917,
            'eight': 3787,
            'five': 4052,
            'follow': 1579,
            'forward': 1557,
            'four': 3728,
            'go': 3880,
            'happy': 2054,
            'House': 2113,
            'Learn': 1575,
            'Left': 3801,
            'Marvin': 2100,
            'Nine': 3934,
            'No': 3941,
            'Off': 3745,
            'On': 3845,
            'One': 3890,
            'Right': 3778,
            'Seven': 3998,
            'Sheila': 2022,
            'Six': 3860,
            'Stop': 3872,
            'Three': 3727,
            'Tree': 1759,
            'Two': 3880,
            'Up': 3723,
            'Visual': 1592,
            'Wow': 2123,
            'Yes': 4044,
            'Zero': 4052,
            }
    plt.pie(data.values(), labels=data.keys())
    # plt.bar(data.keys(), data.values())
    # plt.xticks(rotation=90)
    plt.show()

