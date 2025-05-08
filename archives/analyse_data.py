import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données d'Apple
apple_data = pd.read_csv("AAPL_historical_data.csv", parse_dates=["Date"], index_col="Date")

# Calcul des indicateurs techniques
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

apple_data['RSI'] = calculate_rsi(apple_data)
apple_data['MA50'] = apple_data['Close'].rolling(window=50).mean()
apple_data['MA200'] = apple_data['Close'].rolling(window=200).mean()
ema12 = apple_data['Close'].ewm(span=12, adjust=False).mean()
ema26 = apple_data['Close'].ewm(span=26, adjust=False).mean()
apple_data['MACD'] = ema12 - ema26

# Liste des colonnes pour l'analyse
columns = ['Close', 'RSI', 'MA50', 'MA200', 'MACD']

# Matrice de corrélation
corr_matrix = apple_data[columns].corr()

# Affichage de la matrice de corrélation
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice de corrélation des indicateurs techniques")
plt.show()


plt.scatter(apple_data['RSI'], apple_data['Close'])
plt.title("Corrélation entre RSI et cours de clôture")
plt.xlabel("RSI")
plt.ylabel("Cours de clôture (Close)")
plt.show()


plt.scatter(apple_data['MA50'], apple_data['Close'])
plt.title("Corrélation entre MA50 et cours de clôture")
plt.xlabel("MA50")
plt.ylabel("Cours de clôture (Close)")
plt.show()


# Pondération des indicateurs (valeurs normalisées)
apple_data['Indicator_Score'] = (
    0.4 * (apple_data['RSI'] / 100) +  # RSI normalisé
    0.3 * (apple_data['MA50'] / apple_data['Close'].max()) +  # MA50 normalisée
    0.3 * (apple_data['MACD'] / apple_data['MACD'].max())  # MACD normalisée
)

# Visualisation de l'indicateur croisé
plt.plot(apple_data.index, apple_data['Indicator_Score'], label="Score combiné")
plt.title("Indicateur croisé basé sur RSI, MA50 et MACD")
plt.xlabel("Date")
plt.ylabel("Score")
plt.legend()
plt.show()


plt.scatter(apple_data['Indicator_Score'], apple_data['Close'])
plt.title("Corrélation entre l'indicateur croisé et le cours de clôture")
plt.xlabel("Score combiné")
plt.ylabel("Cours de clôture")
plt.show()