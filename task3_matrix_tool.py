"""
Task 3: Matrix Operations Tool (Python + NumPy)
Interactive menu: addition, subtraction, multiplication, transpose, determinant.
Run:  python task3_matrix_tool.py
"""
import numpy as np


def read_matrix(name):
    """Ask the user for a matrix; returns a NumPy array."""
    while True:
        try:
            r = int(input(f"Rows of matrix {name}: "))
            c = int(input(f"Columns of matrix {name}: "))
            if r <= 0 or c <= 0:
                raise ValueError
            break
        except ValueError:
            print("  Please enter positive whole numbers.")
    print(f"Enter {r} rows, each with {c} numbers separated by spaces:")
    rows = []
    while len(rows) < r:
        try:
            vals = [float(v) for v in input(f"  Row {len(rows) + 1}: ").split()]
            if len(vals) != c:
                print(f"  Need exactly {c} numbers.")
                continue
            rows.append(vals)
        except ValueError:
            print("  Invalid number, try again.")
    return np.array(rows)


def show(title, m):
    """Display a matrix in a bordered, aligned table."""
    m = np.atleast_2d(m)
    cells = [[f"{v:.4g}" for v in row] for row in m]
    w = max(len(c) for row in cells for c in row) + 2
    line = "+" + "+".join(["-" * w] * m.shape[1]) + "+"
    print(f"\n{title}  (shape {m.shape[0]}x{m.shape[1]})")
    print(line)
    for row in cells:
        print("|" + "|".join(c.center(w) for c in row) + "|")
        print(line)


def add(a, b):
    if a.shape != b.shape:
        raise ValueError(f"Shapes must match for addition: {a.shape} vs {b.shape}")
    return a + b


def subtract(a, b):
    if a.shape != b.shape:
        raise ValueError(f"Shapes must match for subtraction: {a.shape} vs {b.shape}")
    return a - b


def multiply(a, b):
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Columns of A ({a.shape[1]}) must equal rows of B ({b.shape[0]})")
    return a @ b


def determinant(a):
    if a.shape[0] != a.shape[1]:
        raise ValueError("Determinant needs a square matrix")
    return np.linalg.det(a)


MENU = """
================ MATRIX OPERATIONS TOOL ================
 1. Addition (A + B)
 2. Subtraction (A - B)
 3. Multiplication (A x B)
 4. Transpose
 5. Determinant
 6. Exit
========================================================"""


def main():
    while True:
        print(MENU)
        choice = input("Choose an option (1-6): ").strip()
        if choice == "6":
            print("Goodbye!")
            break
        if choice not in {"1", "2", "3", "4", "5"}:
            print("Invalid choice.")
            continue
        try:
            if choice in {"1", "2", "3"}:
                a, b = read_matrix("A"), read_matrix("B")
                show("Matrix A", a)
                show("Matrix B", b)
                op = {"1": add, "2": subtract, "3": multiply}[choice]
                name = {"1": "A + B", "2": "A - B", "3": "A x B"}[choice]
                show(f"Result: {name}", op(a, b))
            else:
                a = read_matrix("A")
                show("Matrix A", a)
                if choice == "4":
                    show("Result: Transpose of A", a.T)
                else:
                    print(f"\nDeterminant of A = {determinant(a):.4f}")
        except ValueError as e:
            print(f"\n  Error: {e}")


if __name__ == "__main__":
    main()
