# Data directory

`motor_vibration.csv` is generated from the deterministic synthetic signal simulator. It is not a field recording and is not collected from real motors. Regenerate it with:

```bash
vibeguard generate-data --output data/motor_vibration.csv --samples-per-class 1000 --seed 42
```

The file contains sample identifiers, RPM, sample rate, feature schema version, 20 ordered engineered features, and a synthetic class label. Do not treat generated rows as independent physical assets or use them to substantiate real-world performance.
