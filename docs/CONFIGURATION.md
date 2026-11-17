# Configuration

Config is a JSON file loaded by ScoringConfig.from_json. Three presets ship
in config/. Fields with aliases accept legacy names (e.g.
minimum_face_width).

Category bands
key	meaning
good_threshold	score >= this AND no review-gates => GOOD
review_threshold	score >= this => REVIEW, otherwise BAD
Must satisfy 0 <= review_threshold <= good_threshold <= 100.

Metric weights
Keys must be the six default metric names. Values are normalized on load.
Setting a weight to 0 removes the metric from the sum entirely.

Rule severities
rules maps a rule name to off | review | reject.

off — rule never fires

review — forces at most REVIEW (score still counts toward band)

reject — forces BAD, overriding any score

Rule names are listed in engine.DEFAULT_RULES. All sixteen must be present.

Thresholds
Every numeric field is validated on load; out-of-range values raise
ValueError with a message naming the field.

Calibration required
The shipped thresholds are defaults, not calibrated values. Run
docs/MODELS.md step "Calibration" before trusting the output.