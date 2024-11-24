import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd


t0, t_end = 0, 1
x0 = 0.5  # Начальное условие
h = 0.001  # Шаг дискретизации
t_values = np.arange(t0, t_end + h, h)


# Аналитическое решение (находим его численно с помощью solve_ivp для точности)
def ode_analytical(t, x):
    return -1 + x ** 2


sol_analytical = solve_ivp(ode_analytical, [t0, t_end], [x0], t_eval=t_values)


# Методы численного интегрирования
def euler_method(f, x0, t_values, h):
    x = np.zeros(len(t_values))
    x[0] = x0
    for i in range(1, len(t_values)):
        x[i] = x[i - 1] + h * f(t_values[i - 1], x[i - 1])
    return x


def trapezoidal_method(f, x0, t_values, h):
    x = np.zeros(len(t_values))
    x[0] = x0
    for i in range(1, len(t_values)):
        t, x_prev = t_values[i - 1], x[i - 1]
        x_predict = x_prev + h * f(t, x_prev)
        x[i] = x_prev + (h / 2) * (f(t, x_prev) + f(t_values[i], x_predict))
    return x


def rectangle_method(f, x0, t_values, h):
    x = np.zeros(len(t_values))
    x[0] = x0
    for i in range(1, len(t_values)):
        # Метод правых прямоугольников: используем значение на конце шага
        x[i] = x[i - 1] + h * f(t_values[i], x[i - 1] + h * f(t_values[i - 1], x[i - 1]))
    return x


# Получаем решения для каждого метода
x_analytical = sol_analytical.y[0]
x_euler = euler_method(ode_analytical, x0, t_values, h)
x_trapezoidal = trapezoidal_method(ode_analytical, x0, t_values, h)
x_rectangle = rectangle_method(ode_analytical, x0, t_values, h)

# Построение графиков для сравнения
plt.figure(figsize=(10, 6))
plt.plot(t_values, x_analytical, label="Analytical", color="black")
plt.plot(t_values, x_euler, label="Euler Method", linestyle="--", color="blue")
plt.plot(t_values, x_trapezoidal, label="Trapezoidal Method", linestyle="--", color="green")
plt.plot(t_values, x_rectangle, label="Rectangle Method", linestyle="--", color="red")

plt.xlabel("Time t")
plt.ylabel("x(t)")
plt.title("Comparison of Analytical and Numerical Solutions")
plt.legend(loc="best")
plt.grid(True)
plt.show()
# Вычисление абсолютных ошибок для каждого метода
error_euler = np.abs(x_analytical - x_euler)
error_trapezoidal = np.abs(x_analytical - x_trapezoidal)
error_rectangle = np.abs(x_analytical - x_rectangle)

# Максимальная погрешность для каждого метода
max_error_euler = np.max(error_euler)
max_error_trapezoidal = np.max(error_trapezoidal)
max_error_rectangle = np.max(error_rectangle)

print("Максимальная погрешность для метода Эйлера:", max_error_euler)
print("Максимальная погрешность для метода трапеций:", max_error_trapezoidal)
print("Максимальная погрешность для метода прямоугольников:", max_error_rectangle)

# Создание таблицы значений
table_data = {
    "Time": t_values,
    "Analytical": x_analytical,
    "Эйлера": x_euler,
    "Трапеций": x_trapezoidal,
    "Прямоугольников": x_rectangle
}
comparison_table = pd.DataFrame(table_data)

# Вывод первых 10 строк таблицы
print(comparison_table.head(1000))
