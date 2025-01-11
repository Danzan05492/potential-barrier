import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Константы
hbar = 1.0  # Приведённая постоянная Планка
m = 1.0     # Масса частицы
V0 = 3.0    # Высота потенциального барьера
a = 2.0     # Ширина потенциального барьера

# Пространственная сетка
x = np.linspace(-5, 5, 1000)
dx = x[1] - x[0]

# Потенциальный барьер
V = np.where((x >= -a/2) & (x <= a/2), V0, 0)

# Функция для вычисления волновой функции
def calculate_wave_function(E, V, x, dx):
    """
    Численное решение уравнения Шрёдингера для заданного потенциала.
    """
    psi = np.zeros(len(x), dtype=complex)
    psi[1] = 1e-10  # Начальное малое значение для предотвращения деления на ноль

    for i in range(1, len(x) - 1):
        psi[i + 1] = (2 * (1 - 0.5 * (2 * m / hbar**2) * (E - V[i]) * dx**2) * psi[i]
                      - psi[i - 1])

    return psi

# Энергии для анимации
energies = np.linspace(0.1, 3.5, 200)

# Функция для вычисления коэффициента туннелирования
def calculate_tunneling_coefficient(E, V0, a):
    if E < V0:
        kappa = np.sqrt(2 * m * (V0 - E)) / hbar
        return np.exp(-2 * kappa * a)
    else:
        return 1.0

# Подготовка графика для анимации
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot([], [], color="blue", label="|ψ(x)|² (Плотность вероятности)")
ax.plot(x, V, color="orange", linewidth=2, label="Потенциальный барьер (V)")
ax.axvline(-a/2, color='green', linestyle='--', label="Границы барьера")
ax.axvline(a/2, color='green', linestyle='--')
ax.set_xlim(-5, 5)
ax.set_ylim(0, 1.1 * V0)
ax.set_title("Эволюция волновой функции через потенциальный барьер")
ax.set_xlabel("Позиция (x)")
ax.set_ylabel("Потенциал / Плотность вероятности")
ax.legend()
ax.grid()

# Функция для обновления анимации
def animate(frame):
    E = energies[frame]
    psi = calculate_wave_function(E, V, x, dx)
    psi_squared = np.abs(psi)**2
    psi_squared /= np.max(psi_squared)  # Нормировка для видимости
    line.set_data(x, psi_squared)
    T = calculate_tunneling_coefficient(E, V0, a)
    ax.set_title(f"Волновая функция для энергии E = {E:.2f}, Коэффициент туннелирования T = {T:.3e}")
    return line,

# Создание анимации
ani = animation.FuncAnimation(fig, animate, frames=len(energies), interval=100, blit=True)

# Сохранение анимации в файл
ani.save("wave_function_animation.gif", writer="pillow", fps=10)

# Построение графика зависимости T(E)
tunneling_coefficients = [calculate_tunneling_coefficient(E, V0, a) for E in energies]

plt.figure(figsize=(10, 6))
plt.plot(energies, tunneling_coefficients, color='blue', label="Коэффициент туннелирования T(E)")
plt.axvline(V0, color='orange', linestyle='--', label=f"Высота барьера V₀ = {V0}")
plt.title("Зависимость коэффициента туннелирования от энергии")
plt.xlabel("Энергия частицы (E)")
plt.ylabel("Коэффициент туннелирования (T)")
plt.legend()
plt.grid()
plt.show()
