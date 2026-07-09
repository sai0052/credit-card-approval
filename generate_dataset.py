import pandas as pd

cols = ['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13','A14','A15','A16']
df = pd.read_csv('/home/claude/extracted/crx.data', header=None, names=cols, na_values='?')

out = pd.DataFrame()

# A1 Gender: b -> male(1), a -> female(0)
out['Gender'] = df['A1'].map({'b': 1, 'a': 0})

# A2 Age
out['Age'] = df['A2']

# A3 Debt
out['Debt'] = df['A3']

# A4 Married: u -> yes(1), y/l -> no(0)
out['Married'] = df['A4'].map({'u': 1, 'y': 0, 'l': 0})

# A5 BankCustomer: g -> yes(1), p/gg -> no(0)
out['BankCustomer'] = df['A5'].map({'g': 1, 'p': 0, 'gg': 0})

# A6 Industry: anonymized codes mapped onto app's 8 industry buckets (arbitrary, since
# the UCI dataset scrubbed real category names -- documented as an assumption)
industry_map = {
    'c': 0, 'm': 0,       # other
    'q': 1, 'x': 1,       # it
    'w': 2, 'd': 2,       # banking
    'i': 3, 'e': 3,       # healthcare
    'aa': 4, 'j': 4,      # education
    'ff': 5, 'r': 5,      # government
    'k': 6,               # business
    'cc': 7,              # manufacturing
}
out['Industry'] = df['A6'].map(industry_map)

# A8 YearsEmployed
out['YearsEmployed'] = df['A8']

# A10 EmploymentStatus: t -> employed, f -> unemployed
out['EmploymentStatus'] = df['A10'].map({'t': 'employed', 'f': 'unemployed'})

# A13 Citizenship: g -> indian, p -> nri, s -> other
out['Citizenship'] = df['A13'].map({'g': 'indian', 'p': 'nri', 's': 'other'})

# A9 PriorDefault: t -> no prior default (1, good), f -> prior default (0, bad)
# (t correlates strongly with approval in the raw data, confirming this direction)
out['PriorDefault'] = df['A9'].map({'t': 1, 'f': 0})

# A11 CreditScore (raw 0-67 scale, training script rescales to 300-800)
out['CreditScore'] = df['A11']

# A15 Income
out['Income'] = df['A15']

# A16 Approved: + -> 1, - -> 0
out['Approved'] = df['A16'].map({'+': 1, '-': 0})

out = out.dropna()
out.to_csv('/home/claude/project/cleaned_dataset.csv', index=False)
print(out.shape)
print(out.head(10).to_string())
print(out['Approved'].value_counts())