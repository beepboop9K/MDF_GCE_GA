import sys

def percentage_difference(a, b):
    # Calculate the absolute difference
    diff = abs(a - b)
    # Calculate the average of the two numbers
    avg = (a + b) / 2
    # Calculate the percentage difference
    percent_diff = (diff / avg) * 100
    return percent_diff

if __name__ == "__main__":
    # Ensure the correct number of arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python percentagediff.py [A] [B]")
        sys.exit(1)

    try:
        # Convert arguments to float
        a = float(sys.argv[1])
        b = float(sys.argv[2])

        # Calculate the percentage difference
        result = percentage_difference(a, b)

        # Output the result
        print(f"The percentage difference between {a} and {b} is {result:.2f}%")
    except ValueError:
        print("Error: Please provide valid numbers for A and B.")
        sys.exit(1)

