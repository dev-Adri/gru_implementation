import random

data = "signal1, signal2, signal3, quaternion\n\n"

print("generating ...")
for i in range (100000):
    sig1 = random.random()
    sig2 = random.random()
    sig3 = random.random()

    quat = str([random.random() for _ in range(4)]).replace(" ", "")

    data += f"{sig1};{sig2};{sig3};{quat}\n"

with open("train_data.csv", "w") as f:
    f.write(data)
    print("done generating")