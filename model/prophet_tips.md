# Prophet Tips

## Cross-validation with Prophet

Example Explanation

Let’s assume your dataset has 1500 days of data. Here’s how the cross-validation process would work step-by-step:

1. First Training Set:
   - Training Data: Day 1 to Day 780
   - Validation Data: Day 781 to Day 1145 (365 days into the future)

2. First Cutoff:
   - Move forward by 180 days.
   - Training Data: Day 1 to Day 960
   - Validation Data: Day 961 to Day 1325

3. Second Cutoff:
   - Move forward another 180 days.
   - Training Data: Day 1 to Day 1140
   - Validation Data: Day 1141 to Day 1500

Since the dataset only has 1500 days, the process would stop here because there isn’t enough data to make another full 365-day forecast from the next cutoff point.

Visual Representation

```bash
Here’s a visual representation of how the data slicing looks:
|----initial (780 days)----|----horizon (365 days)----|
|  training                |  validation              |

|----initial (780 days)-------|----period (180 days)----|----horizon (365 days)----|
|         training                                      |  validation (1)          |

|----initial (780 days)----------------|----period (360 days)----|----horizon (365 days)----|
|                 training                                       |  validation (2)        |
```

And so on, until there are no more days left to form a new horizon period.
