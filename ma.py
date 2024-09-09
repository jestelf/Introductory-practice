import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

numbers = [random.randint(-10000, 10000) for _ in range(1000)]

df = pd.DataFrame(numbers, columns=['Numbers'])

print("\n--- Часть 1: Элементарный уровень. Вычисление статистических характеристик и построение графиков ---")

print("Минимальное значение: ", df['Numbers'].min())
print("Количество повторяющихся значений: ", df['Numbers'].duplicated().sum())
print("Максимальное значение: ", df['Numbers'].max())
print("Сумма чисел: ", df['Numbers'].sum())
print("Среднеквадратическое отклонение: ", df['Numbers'].std())

df_rounded = df['Numbers'].round()

plt.figure(figsize=(10, 6))
plt.plot(df['Numbers'])
plt.title('Линейный график')
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(df_rounded, bins=30)
plt.title('Гистограмма')
plt.show()

df_sorted_asc = df.sort_values(by='Numbers')
plt.figure(figsize=(10, 6))
plt.plot(df_sorted_asc['Numbers'].reset_index(drop=True))
plt.title('Линейный график отсортированных значений по возрастанию')
plt.show()

df_sorted_desc = df.sort_values(by='Numbers', ascending=False)
plt.figure(figsize=(10, 6))
plt.plot(df_sorted_desc['Numbers'].reset_index(drop=True))
plt.title('Линейный график отсортированных значений по убыванию')
plt.show()

id = 70188456
sum_id = sum(int(digit) for digit in str(id))

print("\n--- Часть 2: Базовый уровень. Работа с матрицей и вектором, основанными на ID ---")
print(f"ID: {id}, Сумма цифр ID: {sum_id}")

array = np.random.randint(low=-100, high=100, size=(sum_id, sum_id))

try:
    det = np.linalg.det(array)
    print("Определитель матрицы: ", det)
except np.linalg.LinAlgError:
    print("Определитель не может быть вычислен")

B = np.random.randint(low=-100, high=100, size=(sum_id, 1))

try:
    X = np.linalg.solve(array, B)
    print("\nРешение матричного уравнения A*X = B:")
    print(X)
except np.linalg.LinAlgError:
    print("Матричное уравнение не может быть решено")

mean_values = np.mean(array, axis=0)
print("\nСреднее значение каждого столбца:")
print(mean_values)

sum_values = np.sum(array, axis=1)
print("\nСумма чисел каждой строки:")
print(sum_values)

df_result = pd.DataFrame()

for idx, row in enumerate(array):
    df_result[f'Строка массива {idx+1}'] = pd.Series(row)

df_result['Среднее значение каждого столбца'] = pd.Series(mean_values)
df_result['Сумма чисел каждой строки'] = pd.Series(sum_values)

df_result['Вектор-столбец B'] = pd.Series(B.flatten())

try:
    det = np.linalg.det(array)
    X = np.linalg.solve(array, B)
    df_result['Определитель матрицы'] = pd.Series([det])
    df_result['Решение матричного уравнения'] = pd.Series([X.flatten()])
except np.linalg.LinAlgError:
    print("Определитель и решение матричного уравнения не могут быть вычислены")

df_result.to_csv('result.csv', index=False)
print("\nРезультаты сохранены в файл 'result.csv'.")

print("\n--- Часть 3: Продвинутый уровень. Анализ 9 датасетов с последующей сортировкой ---")

datasets = {}
for file_name in os.listdir():
    if file_name.endswith('.csv'):
        datasets[file_name] = pd.read_csv(file_name)

for name, df in datasets.items():
    print(f"Датасет: {name}")
    print(f"Количество элементов: {len(df)}")
    print()

order_items = pd.read_csv('olist_order_items_dataset.csv')
orders = pd.read_csv('olist_orders_dataset.csv')
products = pd.read_csv('olist_products_dataset.csv')
customers = datasets['olist_customers_dataset.csv']

order_items = pd.merge(order_items, products, on='product_id')

print(f"Количество продуктов: {len(products)}")
print(f"Средняя цена товара по категориям:\n{order_items.groupby('product_category_name')['price'].mean()}")
print(f"Максимальная цена товара по категориям:\n{order_items.groupby('product_category_name')['price'].max()}")

order_items = order_items.merge(orders[['order_id', 'customer_id', 'order_status']], on='order_id', how='left')
order_items = order_items.merge(customers[['customer_id', 'customer_state']], on='customer_id', how='left')

delivered_orders = order_items[order_items['order_status'] == 'delivered']
purchases_by_category = delivered_orders.groupby('product_category_name')['order_id'].count()
purchases_by_state = delivered_orders.groupby('customer_state')['order_id'].count()
total_purchases = len(delivered_orders)
print(f"Процент покупок по категориям:\n{purchases_by_category / total_purchases * 100}")
print(f"Процент закупок государством:\n{purchases_by_state / total_purchases * 100}")

products['description_length_bucket'] = (products['product_description_lenght'] // 25) * 25
avg_purchases_per_description_length = order_items.groupby(products['description_length_bucket'])['order_id'].count() / products.groupby('description_length_bucket')['product_id'].count()
plt.plot(avg_purchases_per_description_length.index, avg_purchases_per_description_length.values)
plt.xlabel('Длина описания')
plt.ylabel('Среднее количество покупок')
plt.title('Зависимость между средним количеством покупок и длиной описания')
plt.show()

products['name_length_bucket'] = (products['product_name_lenght'] // 10) * 10
avg_purchases_per_name_length = order_items.groupby(products['name_length_bucket'])['order_id'].count() / products.groupby('name_length_bucket')['product_id'].count()
plt.plot(avg_purchases_per_name_length.index, avg_purchases_per_name_length.values)
plt.xlabel('Длина имени')
plt.ylabel('Среднее количество покупок')
plt.title('Зависимость между средним количеством покупок и длиной имени')
plt.show()

sellers = datasets['olist_sellers_dataset.csv']
order_items = order_items.merge(sellers[['seller_id', 'seller_state']], on='seller_id', how='left')
sellers['sells_to_other_regions'] = sellers['seller_id'].isin(order_items[order_items['customer_state'] != order_items['seller_state']]['seller_id'])
top_sellers = sellers[sellers['sells_to_other_regions']].groupby('seller_id')['seller_id'].count().nlargest(5)
print(f'Топ-5 продавцов, которые часто отправляют свои посылки в другие регионы: {top_sellers.index.tolist()}')

products_info = pd.DataFrame({
    "Средняя цена товара по категориям": order_items.groupby('product_category_name')['price'].mean(),
    "Максимальная цена товара по категориям": order_items.groupby('product_category_name')['price'].max()
})
products_info.to_csv('products_info.csv', index=True)

purchases_info = pd.DataFrame({
    "Процент покупок по категориям": purchases_by_category / total_purchases * 100,
    "Процент закупок государством": purchases_by_state / total_purchases * 100
})
purchases_info.to_csv('purchases_info.csv', index=True)

top_sellers_df = pd.DataFrame(top_sellers)
top_sellers_df.columns = ["Число продаж"]
top_sellers_df.to_csv('top_sellers_info.csv', index=True)
