import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return np.exp(np.cos(x))


def midpoint_rule(fcn, a, b, n):
    Deltax = (b - a) / n
    xs = [a + Deltax * i for i in range(n + 1)]
    midpoints = [(xs[i] + xs[i + 1]) / 2 for i in range(n)]
    ysmid = [fcn(mp) for mp in midpoints]
    return Deltax * sum(ysmid), xs, midpoints, ysmid


# Exact value of integral from 0 to 1 of sin(x)
exact_value = 2 / 7

# Create sequence of n values (halving each time)
n = 512
n_intervals = []
current_n = n
while current_n >= 2:
    n_intervals.append(current_n)
    current_n = current_n // 2

# Compute results
results = []
for n in n_intervals:
    approx_value, xs, midpoints, ysmid = midpoint_rule(f, 0, np.divide(np.pi, 4), n)
    error = abs(exact_value - approx_value)
    if len(results) > 0:
        ratio = results[-1]["Error"] / error
    else:
        ratio = np.nan
    results.append(
        {
            "n": n,
            "Approximation": approx_value,
            "Exact": exact_value,
            "Error": error,
            "Error Ratio": ratio,
        }
    )

# Create DataFrame
df = pd.DataFrame(results)
df["Error Ratio"] = df["Error Ratio"].apply(
    lambda x: f"{x:.2f}" if not np.isnan(x) else ""
)

# Save DataFrame as image
plt.figure(figsize=(10, 4))
plt.axis("off")
plt.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center")
plt.title("Midpoint Rule Approximation Results", pad=20)
plt.savefig("midpoint_rule_results.png", bbox_inches="tight", dpi=300)
plt.close()

# Create and save plot visualization
n = 512  # Use largest n for visualization
approx_value, xs, midpoints, ysmid = midpoint_rule(f, 0, np.divide(np.pi, 4), n)

x_vals = np.linspace(0, np.divide(np.pi, 4), n)
y_vals = f(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, "b-", label="e^(cos(x))")

# Draw rectangles
for i in range(n):
    left = xs[i]
    right = xs[i + 1]
    midpoint = midpoints[i]
    height = ysmid[i]
    plt.bar(
        midpoint,
        height,
        width=(right - left),
        align="center",
        alpha=0.3,
        edgecolor="red",
        color="orange",
    )
    plt.plot([midpoint, midpoint], [0, height], "r-")

plt.title(
    f"Midpoint Rule Approximation (n={n})\nApproximation: {approx_value:.6f} | Exact: {exact_value:.6f} | Error: {abs(exact_value - approx_value):.2e}"
)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.savefig("midpoint_rule_visualization.png", bbox_inches="tight", dpi=300)
plt.close()

print("Results saved to midpoint_rule_results.png and midpoint_rule_visualization.png")
