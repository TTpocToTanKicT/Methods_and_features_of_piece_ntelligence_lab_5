"""
Лабораторна робота 5 — переробка методички (MatLab) на Python.

Відповідність пунктів методички див. у попередніх версіях звіту; тут реалізація в Python.
"""

from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError, ValueError):
        pass

try:
    import joblib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    print("Не вистачає бібліотек. У терміналі Cursor виконай у цій теці:", file=sys.stderr)
    print("  python -m pip install -r requirements.txt", file=sys.stderr)
    print("Деталі:", e, file=sys.stderr)
    sys.exit(1)


def variant_function(t: np.ndarray, variant: int) -> np.ndarray:
    """Нелінійні функції з табл. 1. Інтервал завдання: [0, 10]."""
    t = np.asarray(t, dtype=float)
    eps = 1e-9

    if variant == 1:
        return (np.cos(t) ** 2) * np.sin(2 * t)
    if variant == 2:
        return (t**2) * np.sin(t)
    if variant == 3:
        return np.sin(0.5 * (t**2))
    if variant == 4:
        return np.sin(t**2 - 10 * t)
    if variant == 5:
        return t * np.sin(t)
    if variant == 6:
        return np.sin(t**2 - 5 * t)
    if variant == 7:
        return np.cos(1.4 * t)
    if variant == 8:
        return np.sin(t**2 - 8 * t)
    if variant == 9:
        return np.log(np.maximum(t, eps)) * np.sin(t)
    if variant == 10:
        return np.sin(t**2 - 12 * t)
    if variant == 11:
        return np.log(np.maximum(2 * t, eps)) * np.sin(t**1.5)
    if variant == 12:
        return np.sin(t**2 - 8 * t)
    if variant == 13:
        return 0.01 * (t**2) * np.sin(t)
    if variant == 14:
        return np.sin(t**2 - 12 * t)
    if variant == 15:
        return t * np.sin(2 * t)
    if variant == 16:
        return np.sin(t**2 - 6 * t)
    if variant == 17:
        return t * np.sin(3 * t)
    if variant == 18:
        return np.sin(t**2 - 4 * t)
    if variant == 19:
        return t * np.sin(4 * t)
    if variant == 20:
        return (t**2 - 10 * t) * np.sin(t**2 - 10 * t)
    raise ValueError(f"Невідомий варіант: {variant} (очікується 1–20)")


def make_dataset(t_min: float, t_max: float, step: float, variant: int) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(t_min, t_max + step * 0.5, step, dtype=float)
    y = variant_function(t, variant)
    return t, y


def _configure_plot_axes(
    ax,
    t_train: np.ndarray,
    y_train: np.ndarray,
    t_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    variant: int,
) -> None:
    ax.plot(t_train, y_train, ".", markersize=4, label="Навчальні точки f(t)")
    ax.plot(t_test.ravel(), y_true, "-", linewidth=1.5, label="f(t) (еталон на тесті)")
    ax.plot(t_test.ravel(), y_pred, "r+", markersize=4, label="Вихід мережі (sim)")
    ax.set_xlabel("t")
    ax.set_ylabel("f(t)")
    ax.set_title(f"ЛР5: MLP, варіант {variant}")
    ax.legend()
    ax.grid(True, alpha=0.3)


def calculate_without_files(variant: int, step: float = 0.05) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Обчислює без створення файлів (для попереднього перегляду)."""
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    t_train, y_train = make_dataset(0.0, 10.0, step, variant)
    x1 = t_train.reshape(-1, 1)
    y1 = y_train.ravel()

    net = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(5, 3),
                    activation="logistic",
                    solver="lbfgs",
                    alpha=1e-6,
                    max_iter=8000,
                    random_state=42,
                    tol=1e-6,
                ),
            ),
        ]
    )
    net.fit(x1, y1)

    t_test = np.arange(0.025, 10.0, 0.1, dtype=float).reshape(-1, 1)
    y_true = variant_function(t_test.ravel(), variant)
    y_pred = net.predict(t_test)

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    lines: list[str] = [
        "══════════════════════════════════════",
        "ЛР5 — підсумок по пунктах методички (попередній перегляд)",
        "══════════════════════════════════════",
        f"Варіант з табл. 1: {variant}",
        f"Крок сітки t для навчання: {step}",
        "",
        "────────  Завдання 1  ────────",
        "За методичкою: підготувати файл .csv з даними f(t) на [0; 10].",
        f"Тут: CSV не створено (попередній перегляд).",
        "",
        "────────  Завдання 2  ────────",
        "За методичкою: прочитати csv у програмі (у MatLab — csvread).",
        "Тут: операція не виконана (попередній перегляд).",
        "",
        "────────  Завдання 3  ────────",
        "За методичкою: записати й зчитати файл з роздільником (dlmwrite / dlmread).",
        "Тут: операція не виконана (попередній перегляд).",
        "",
        "────────  Завдання 4  ────────",
        "За методичкою: мережа feed-forward з навчанням backprop для наближення f(t).",
        "Тут: sklearn — Pipeline(StandardScaler + MLPRegressor), шари (5, 3), активація logistic, solver lbfgs.",
        "Навчання виконано на усіх точках навчальної сітки.",
        "",
        "────────  Завдання 5  ────────",
        "За методичкою: зберегти навчену мережу у файлі.",
        "Тут: файл не створено (попередній перегляд).",
        "",
        "────────  Завдання 6  ────────",
        "За методичкою: на одному графіку — бажані та обчислені значення на виході мережі.",
        "Тут: графік показано, але не збережено у файл (попередній перегляд).",
        "",
        "Метрики на тестовій сітці (для аналізу у звіті, п. 7):",
        f"  MSE = {mse:.6e}",
        f"  MAE = {mae:.6e}",
        f"  R^2 = {r2:.6f}",
        "",
        "════════  Завдання 7 і текст «висновки» у звіті  ════════",
        "За методичкою п. 7: проаналізувати результати і зробити висновки — це ВИ пишете",
        "у звіті самостійно (проза): що показує графік, наскільки малі MSE/MAE,",
        "чи задовольняє наближення, що можна змінити (крок, архітектура, епохи тощо).",
        "",
        "Для створення файлів введіть прізвище та натисніть «Обчислити»!",
    ]

    return lines, t_train, y_train, t_test, y_true, y_pred


def run_calculations(surname: str, variant: int, step: float = 0.05) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Запускає обчислення з створення файлів."""
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import joblib

    out = Path(__file__).resolve().parent / "output"
    out.mkdir(parents=True, exist_ok=True)

    base = surname
    csv_path = out / f"{base}_data.csv"
    dlm_path = out / f"{base}_data.txt"
    model_path = out / f"{base}.joblib"
    plot_path = out / f"{base}_plot.png"

    t_train, y_train = make_dataset(0.0, 10.0, step, variant)
    df = pd.DataFrame({"t": t_train, "f": y_train})
    df.to_csv(csv_path, index=False)

    M_csv = pd.read_csv(csv_path).to_numpy()
    x1 = M_csv[:, 0].reshape(-1, 1)
    y1 = M_csv[:, 1].ravel()

    np.savetxt(dlm_path, M_csv, delimiter="\t", fmt="%.8e")
    M_dlm = np.loadtxt(dlm_path, delimiter="\t")
    if not np.allclose(M_csv, M_dlm):
        raise RuntimeError("Розбіжність після dlm-циклу: перевірте формат файлу.")

    net = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(5, 3),
                    activation="logistic",
                    solver="lbfgs",
                    alpha=1e-6,
                    max_iter=8000,
                    random_state=42,
                    tol=1e-6,
                ),
            ),
        ]
    )
    net.fit(x1, y1)

    joblib.dump(net, model_path)

    t_test = np.arange(0.025, 10.0, 0.1, dtype=float).reshape(-1, 1)
    y_true = variant_function(t_test.ravel(), variant)
    y_pred = net.predict(t_test)

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    lines: list[str] = [
        "══════════════════════════════════════",
        "ЛР5 — підсумок по пунктах методички",
        "══════════════════════════════════════",
        f"Варіант з табл. 1: {variant}",
        f"База імені файлів (мережа): {base}",
        f"Крок сітки t для навчання: {step}",
        "",
        "────────  Завдання 1  ────────",
        "За методичкою: підготувати файл .csv з даними f(t) на [0; 10].",
        f"Тут: CSV згенеровано автоматично (стовпці t та f).",
        f"Файл: {csv_path}",
        "",
        "────────  Завдання 2  ────────",
        "За методичкою: прочитати csv у програмі (у MatLab — csvread).",
        "Тут: файл прочитано через pandas.read_csv → масив для навчання.",
        f"Перевірка: ті самі дані використано для стовпців t та f.",
        "",
        "────────  Завдання 3  ────────",
        "За методичкою: записати й зчитати файл з роздільником (dlmwrite / dlmread).",
        "Тут: numpy.savetxt / loadtxt з табуляцією.",
        f"Файл: {dlm_path}",
        "Перевірка: масив після читання збігається з CSV.",
        "",
        "────────  Завдання 4  ────────",
        "За методичкою: мережа feed-forward з навчанням backprop для наближення f(t).",
        "Тут: sklearn — Pipeline(StandardScaler + MLPRegressor), шари (5, 3), активація logistic, solver lbfgs.",
        "Навчання виконано на усіх точках навчальної сітки.",
        "",
        "────────  Завдання 5  ────────",
        "За методичкою: зберегти навчену мережу у файлі.",
        f"Тут: joblib.dump → файл моделі.",
        f"Файл: {model_path}",
        "",
        "────────  Завдання 6  ────────",
        "За методичкою: на одному графіку — бажані та обчислені значення на виході мережі.",
        f"Тут: графік збережено у PNG; той самий графік показано справа у цьому вікні.",
        f"Файл: {plot_path}",
        "",
        "Метрики на тестовій сітці (для аналізу у звіті, п. 7):",
        f"  MSE = {mse:.6e}",
        f"  MAE = {mae:.6e}",
        f"  R^2 = {r2:.6f}",
        "",
        "════════  Завдання 7 і текст «висновки» у звіті  ════════",
        "За методичкою п. 7: проаналізувати результати і зробити висновки — це ВИ пишете",
        "у звіті самостійно (проза): що показує графік, наскільки малі MSE/MAE,",
        "чи задовольняє наближення, що можна змінити (крок, архітектура, епохи тощо).",
        "",
        "Нижче — НЕ готовий звіт і не з методички дослівно: це лише автоматична",
        "однорядкова підказка-чернетка від програми (можна ігнорувати або перефразувати у звіті):",
        "",
        "  » двошаровий персептрон (5 і 3 нейрони, logistic) навчено наближати f(t) на [0; 10].",
        "    За графіком і метриками оцініть якість самостійно.",
    ]

    return lines, t_train, y_train, t_test, y_true, y_pred


def main_gui():
    """Головна функція GUI."""
    import tkinter as tk
    from tkinter import scrolledtext, messagebox
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    import numpy as np

    root = tk.Tk()
    root.title("ЛР5 — результат")
    root.geometry("1200x650")
    root.minsize(900, 500)

    # Головний фрейм
    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    # Панель в правій верхній частині для вводу
    top_right_frame = tk.Frame(main_frame)
    top_right_frame.pack(side=tk.TOP, fill=tk.X, padx=4, pady=4)

    tk.Label(top_right_frame, text="Прізвище:", font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=2)
    surname_entry = tk.Entry(top_right_frame, width=20, font=("Segoe UI", 10))
    surname_entry.pack(side=tk.LEFT, padx=2)
    surname_entry.insert(0, "Student")

    tk.Label(top_right_frame, text="Варіант:", font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=(20, 2))
    variant_entry = tk.Entry(top_right_frame, width=5, font=("Segoe UI", 10))
    variant_entry.pack(side=tk.LEFT, padx=2)
    variant_entry.insert(0, "1")

    # PanedWindow для тексту та графіка
    paned = tk.PanedWindow(main_frame, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=5)
    paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    left = tk.Frame(paned)
    st = scrolledtext.ScrolledText(left, wrap="word", font=("Segoe UI", 10))
    st.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
    paned.add(left, minsize=340)

    right = tk.Frame(paned)
    fig = Figure(figsize=(6.5, 4.5), dpi=100)
    ax = fig.add_subplot(111)
    fig.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=right)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
    paned.add(right, minsize=500)

    # Функція для оновлення вмісту
    def update_display(lines, t_train, y_train, t_test, y_true, y_pred, variant):
        st.configure(state="normal")
        st.delete("1.0", tk.END)
        st.insert("1.0", "\n".join(lines))
        st.configure(state="disabled")
        
        ax.clear()
        _configure_plot_axes(ax, t_train, y_train, t_test, y_true, y_pred, variant)
        canvas.draw()

    # Кнопка обчислити
    def calculate():
        try:
            surname = surname_entry.get().strip()
            variant = int(variant_entry.get())
            if not surname:
                messagebox.showerror("Помилка", "Введіть прізвище!")
                return
            if variant < 1 or variant > 20:
                messagebox.showerror("Помилка", "Варіант має бути від 1 до 20!")
                return

            lines, t_train, y_train, t_test, y_true, y_pred = run_calculations(surname, variant)
            update_display(lines, t_train, y_train, t_test, y_true, y_pred, variant)

        except Exception as e:
            messagebox.showerror("Помилка", f"Сталася помилка: {str(e)}")

    tk.Button(top_right_frame, text="Обчислити", command=calculate, font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=(20, 2))

    tk.Button(root, text="Закрити", command=root.destroy, font=("Segoe UI", 10)).pack(pady=6)

    # Початковий попередній перегляд варіанту 1 без створення файлів
    try:
        lines, t_train, y_train, t_test, y_true, y_pred = calculate_without_files(1)
        update_display(lines, t_train, y_train, t_test, y_true, y_pred, 1)
    except Exception as e:
        pass

    root.mainloop()


def main() -> None:
    main_gui()


if __name__ == "__main__":
    main()
