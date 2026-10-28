# Fault-detection results

The transparent demonstration classifier correctly identified **100% of the three simulated conditions**.

```text
              Predicted condition  RMS (g)  Peak (g)  Crest factor  Kurtosis  Dominant frequency (Hz)  Envelope peak (Hz)  Resonance energy
Condition                                                                                                                                  
normal                     normal    0.143     0.415         2.896     2.080                     30.0               145.5             0.002
imbalance               imbalance    0.680     1.138         1.674     1.528                     30.0               475.5             0.002
bearing_fault       bearing_fault    0.237     1.311         5.537     7.669                     30.0               120.0             0.049
```

The imbalance case is dominated by the 30 Hz shaft frequency. The bearing-fault case
contains repeated impacts near 120 Hz that excite a 2.5 kHz resonance and become visible
through envelope analysis. Thresholds were designed for these simulations and must not
be treated as validated industrial alarm limits.
