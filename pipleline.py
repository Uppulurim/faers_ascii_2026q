import numpy as np
import pandas as pd

def calc_ror(a, b, c, d):
    if b == 0 or c == 0:
        return None, None, None
    ror = (a * d) / (b * c)
    se = np.sqrt(1/a + 1/b + 1/c + 1/d) if a > 0 else None
    if se is None:
        return ror, None, None
    ci_lower = np.exp(np.log(ror) - 1.96 * se)
    ci_upper = np.exp(np.log(ror) + 1.96 * se)
    return ror, ci_lower, ci_upper

def calc_prr(a, b, c, d):
    if (a + b) == 0 or (c + d) == 0 or c == 0:
        return None, None
    prr = (a / (a + b)) / (c / (c + d))
    chi2 = ((a*d - b*c)**2 * (a+b+c+d)) / ((a+b)*(c+d)*(a+c)*(b+d))
    return prr, chi2

def signal_flag(a, ror_ci_lower, prr, chi2):
    return (a >= 3) and (ror_ci_lower is not None and ror_ci_lower > 1) \
        and (prr is not None and prr >= 2) and (chi2 is not None and chi2 >= 4)

if __name__ == "__main__":
    a, b, c, d = 10, 90, 50, 5000
    ror, ci_l, ci_u = calc_ror(a, b, c, d)
    prr, chi2 = calc_prr(a, b, c, d)
    print(f"ROR={ror:.2f} (95% CI {ci_l:.2f}-{ci_u:.2f})")
    print(f"PRR={prr:.2f}, chi2={chi2:.2f}")
    print(f"Signal flagged: {signal_flag(a, ci_l, prr, chi2)}")

