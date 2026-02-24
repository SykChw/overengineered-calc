import torch
import random


# ---------- Helper Functions ----------

def int_to_padded_tensor(a_int, b_int, device="cpu"):
    """
    Converts two integers into equal-length digit tensors.
    Returns tensors of shape (1, seq_len)
    """
    a_str = str(a_int)
    b_str = str(b_int)

    max_len = max(len(a_str), len(b_str))

    a_str = a_str.zfill(max_len)
    b_str = b_str.zfill(max_len)

    a_tensor = torch.tensor(
        [[int(d) for d in a_str]],
        dtype=torch.float32,
        device=device
    )

    b_tensor = torch.tensor(
        [[int(d) for d in b_str]],
        dtype=torch.float32,
        device=device
    )

    return a_tensor, b_tensor


def tensor_to_int(tensor):
    """
    Converts output digit tensor back to Python int.
    """
    digits = tensor.squeeze(0).tolist()
    digits = [int(round(d)) for d in digits]
    return int("".join(map(str, digits)))


# ---------- Single Test ----------

def test_single(model, a_int, b_int, device="cpu"):
    model.eval()

    a, b = int_to_padded_tensor(a_int, b_int, device)

    with torch.no_grad():
        result_digits = model(a, b)

    result_int = tensor_to_int(result_digits)
    expected = a_int + b_int

    print("A:", a_int)
    print("B:", b_int)
    print("Model:", result_int)
    print("Expected:", expected)

    assert result_int == expected, "❌ Mismatch!"
    print("✅ Correct!\n")


# ---------- Randomized Stress Test ----------

def stress_test(model, num_tests=1000, max_digits=50, device="cpu"):
    model.eval()

    for i in range(num_tests):

        digits = random.randint(1, max_digits)

        a_int = random.randint(0, 10**digits - 1)
        b_int = random.randint(0, 10**digits - 1)

        a, b = int_to_padded_tensor(a_int, b_int, device)

        with torch.no_grad():
            result_digits = model(a, b)

        result_int = tensor_to_int(result_digits)

        if result_int != a_int + b_int:
            print("❌ Failure detected")
            print("A:", a_int)
            print("B:", b_int)
            print("Got:", result_int)
            print("Expected:", a_int + b_int)
            return

    print(f"✅ Passed {num_tests} random tests!")


# ---------- Example Usage ----------

if __name__ == "__main__":

    device = "cpu"

    model = ReLUAdderRNN().to(device)

    # Simple example
    test_single(model, 999999999, 1, device)

    # Large numbers
    test_single(
        model,
        987654321012345678901234567890,
        123456789098765432109876543210,
        device
    )

    # Stress test
    stress_test(model, num_tests=1000, max_digits=100)
