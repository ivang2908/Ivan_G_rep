import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from scipy.interpolate import interp1d

# Определение функции оценки плотности ядра (Kernel Density Estimation, KDE)
def kde(data, probability_fcn="cdf", bandwidth=None):
    # Если ширина полосы не задана, используется значение по умолчанию
    if bandwidth is None:
        bandwidth = 1.0
    # Создание экземпляра KDE с гауссовым ядром и обучение модели
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(data[:, None])
    # Генерация точек для оценки плотности
    x = np.linspace(min(data), max(data), 1000)
    # Расчет логарифмической плотности
    log_dens = kde.score_samples(x[:, None])
    # Преобразование логарифмической плотности в обычную плотность
    density = np.exp(log_dens)
    # Возврат кумулятивной функции распределения, если это требуется
    if probability_fcn == "cdf":
        cdf = np.cumsum(density)
        cdf /= cdf[-1]
        return cdf, x, bandwidth
    else:
        return density, x, bandwidth

# Функция для генерации случайных выборок на основе KDE
def randkde(data, dim):
    # Обработка данных на наличие NaN и бесконечностей
    data = np.nan_to_num(data, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
    # Получение функции распределения из KDE
    F, x, bw = kde(data, probability_fcn="cdf")
    # Гарантия начала и конца интервала значений функции распределения
    F[0] = 0
    F[-1] = 1
    # Интерполяция для создания функции обратного распределения
    inv_cdf = interp1d(F, x, kind='linear', fill_value="extrapolate")
    # Генерация равномерно распределенных случайных чисел
    total_elements = np.prod(dim)
    uniform_random_samples = np.random.rand(total_elements)
    # Преобразование равномерных значений в значения с нужным распределением
    y = inv_cdf(uniform_random_samples).reshape(dim)
    return y, F, x, bw

# Функции для предобработки данных
def filter_infinite_values(data):
    # Замена NaN и бесконечностей на безопасные значения
    return np.nan_to_num(data, nan=0.0, posinf=np.finfo(np.float64).max, neginf=-np.finfo(np.float64).max)


def clamp_values(data, min_value, max_value):
    # Ограничение значений данных заданным диапазоном
    return np.clip(data, min_value, max_value)


def safe_arctanh(x):
    # Безопасное вычисление гиперболического арктангенса
    epsilon = 1e-10
    x = np.clip(x, -1 + epsilon, 1 - epsilon)
    return np.arctanh(x)

# Настройка основных параметров модели
dv = 3
dc = 6
sigma = 0.75

N = 500000
mu = 2 / sigma ** 2
n = dc - 1
m = dv - 1

# Подготовка к сохранению результатов
fname_pref = f'dv={dv}dc={dc}s={sigma}'
CNfile_path = f'{fname_pref}CNhist.csv'
VNfile_path = f'{fname_pref}VNhist.csv'
CNfile = open(CNfile_path, 'w')
VNfile = open(VNfile_path, 'w')

# Настройка фигур для визуализации результатов
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

Lmax = 30 # Максимальное количество итераций
Lplot = np.arange(1, Lmax + 1) # Диапазон итераций для вывода результатов
numLastPlots = 1 # Количество последних графиков для обновления

# Генерация начальных значений
VmNn = np.random.normal(mu, np.sqrt(2 * mu), (m, N, n))
VmNn = filter_infinite_values(VmNn)
VmNn = clamp_values(VmNn, -1000, 1000)
Uarx = np.zeros((numLastPlots, N))
Varx = np.zeros((numLastPlots, N))
for i in range(numLastPlots):
    Varx[i, :] = VmNn[0, :, 0]

tstart = datetime.now() # Начало измерения времени выполнения
tcurr = datetime.now() # Текущее время для отслеживания длительности итераций

# Основной цикл обработки данных
for L in range(1, Lmax + 1):
    if L % 10 == 0:
        # Вывод текущего шага и времени выполнения
        print(L, datetime.now() - tcurr, datetime.now() - tstart)
        tcurr = datetime.now()

    # Преобразование значений с использованием безопасной функции arctanh
    UmN = 2 * safe_arctanh(np.prod(np.tanh(VmNn / 2), axis=2))
    UmN = filter_infinite_values(UmN)
    UmN = clamp_values(UmN, -1000, 1000)
    UmNV = np.minimum(UmN, 1000)
    # Расчет новых значений на основе текущих
    V = np.sum(UmN, axis=0) + np.random.normal(mu, np.sqrt(2 * mu), N)
    V = filter_infinite_values(V)
    V = clamp_values(V, -1000, 1000)

    # Обновление исторических данных для графиков
    for i in range(numLastPlots - 1):
        Varx[i, :] = Varx[i + 1, :]
        Uarx[i, :] = Uarx[i + 1, :]
    Varx[numLastPlots - 1, :] = V
    Uarx[numLastPlots - 1, :] = UmN[0, :]

    # Генерация новых значений с использованием KDE
    VmNn = randkde(V, (m, N, n))[0]
    VmNn = filter_infinite_values(VmNn)
    VmNn = clamp_values(VmNn, -1000, 1000)

    # Проверка условия сходимости
    converged = np.sum(V < 0) <= 0
    for i in range(converged + numLastPlots * int(not converged), numLastPlots + 1):
        l = L + i - numLastPlots
        if l < L and l in Lplot:
            continue

        # Визуализация и запись данных Uarx
        valid_Uarx = Uarx[i - 1, np.isfinite(Uarx[i - 1, :])]
        valid_Varx = Varx[i - 1, np.isfinite(Varx[i - 1, :])]

        ax1.hist(valid_Uarx, bins=50, density=True, histtype='step', linewidth=1, label=f'L={l}')
        ax1.grid(True)
        ax1.set_ylim([0, np.max(np.histogram(valid_Uarx, bins=50, density=True)[0]) * 3])
        x = (np.histogram(valid_Uarx, bins=50)[1][1:] + np.histogram(valid_Uarx, bins=50)[1][:-1]) / 2

        # Запись данных в файлы для каждого шага, содержащие информацию о гистограммах
        CNfile.write('\n'.join(
            [f'{l}; {xi:.3f}; {vi}' for xi, vi in zip(x, np.histogram(valid_Uarx, bins=50, density=True)[0])]) + '\n')

        # Визуализация гистограммы для переменных Varx
        ax2.hist(valid_Varx, bins=50, density=True, histtype='step', linewidth=1, label=f'L={l}')
        ax2.grid(True)
        x = (np.histogram(valid_Varx, bins=50)[1][1:] + np.histogram(valid_Varx, bins=50)[1][:-1]) / 2

        # Запись данных Varx в файл
        VNfile.write('\n'.join(
            [f'{l}; {xi:.3f}; {vi}' for xi, vi in zip(x, np.histogram(valid_Varx, bins=50, density=True)[0])]) + '\n')

        # Условие для специальной обработки на определённых шагах или при сходимости
        if l in Lplot or l == Lmax or converged:
            # Вычисление диапазона значений для создания более плавной кривой распределения
            xmin = np.min(np.histogram(valid_Varx, bins=50)[1])
            xmax = np.max(np.histogram(valid_Varx, bins=50)[1])
            x = np.linspace(xmin, xmax, 1000)
            absx = np.abs(x)
            valid_Varx = valid_Varx[np.isfinite(valid_Varx)]
            if len(valid_Varx) == 0:
                continue
            # Расчёт статистических моментов
            b2 = np.sum(valid_Varx ** 2) / len(valid_Varx)
            b4 = np.sum(valid_Varx ** 4) / len(valid_Varx)
            # Ограничение значений для избежания числовых ошибок
            b2 = clamp_values(b2, -1e10, 1e10)
            b4 = clamp_values(b4, -1e10, 1e10)
            # Вычисление параметров для нормального распределения
            a2 = np.sqrt((3 * b2 ** 2 - b4) / 2 + 1e-10)
            a = np.sqrt(a2)
            b = np.sqrt(b2 - a2 + 1e-10)
            # Визуализация абсолютного нормального распределения
            absVNpdf = norm.pdf(absx, a, b) + norm.pdf(absx, -a, b)
            str_label = f'L={l}, a={a:.3f}, b={b:.3f}, \\neq{np.sqrt(2 * a):.3f}'
            ax2.plot(x, absVNpdf / (1 + np.exp(-x)), label=str_label)
            plt.show()
        else:
            # Очистка графиков, если не требуется специальная обработка
            ax1.cla()
            ax2.cla()
    # Проверка условия сходимости
    if converged:
        break

# Окончание измерения времени и вывод продолжительности работы программы
tend = datetime.now()
print("Duration:", tend - tstart)

# Закрытие файлов данных
CNfile.close()
VNfile.close()

# Настройка заголовков и легенды графиков
ax1.set_title(f'CN hist, d_c={dc}, d_v={dv}, σ={sigma}')
ax1.legend()

ax2.set_title(f'VN hist, d_c={dc}, d_v={dv}, σ={sigma}')
ax2.legend()

fig.savefig(f'{fname_pref}_L={L}.pdf')
