# =======================================
# Load Data
# =======================================
drive.mount('/content/drive')
DATA_DIR = "/content/drive/MyDrive/market"

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")

train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)

print(train.shape, test.shape)
train.head()

