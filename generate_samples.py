import random
import csv

// This script generates data samples to use as seed when one doesn't have an initial set of data to start with

def generate_samples(n=100, output_file="samples.csv"):
    values = [random.random() for _ in range(n)]

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(values)

    print(f"Generated {n} pseudo-random values saved to '{output_file}'")

if __name__ == "__main__":
    generate_samples()
