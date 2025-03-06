import os
import sys
import subprocess
import platform




def create_virtual_environment():
    """Создаёт виртуальное окружение в папке Dacha."""
    # Получаем полный путь к текущему скрипту
    script_path = os.path.realpath(__file__)
    print("Скрипт находится здесь:", script_path)

    # Получаем директорию, в которой лежит скрипт
    script_dir = os.path.dirname(script_path)
    print("Директория скрипта:", )
    dacha_dir = script_dir
    venv_dir = os.path.join(dacha_dir, "venv")

    if not os.path.exists(dacha_dir):
        os.makedirs(dacha_dir)

    if os.path.exists(venv_dir):
        print("Виртуальное окружение уже существует.")
        return venv_dir

    try:
        print("Создаём виртуальное окружение...")
        subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
        print("Виртуальное окружение успешно создано.")
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при создании виртуального окружения: {e}")
        sys.exit(1)

    return venv_dir


def install_requirements(venv_dir):
    """Устанавливает зависимости из requirements.txt."""
    requirements_file = os.path.join(os.path.dirname(venv_dir), "requirements.txt")
    print(f"Ищем файл {requirements_file}")

    if not os.path.exists(requirements_file):
        print(f"Файл {requirements_file} не найден.")
        print("Текущий рабочий каталог:", os.getcwd())
        print("Содержимое папки Dacha:", os.listdir(os.path.dirname(venv_dir)))
        sys.exit(1)

    pip_executable = os.path.join(
        venv_dir, "Scripts" if platform.system() == "Windows" else "bin", "pip"
    )
    if platform.system() == "Windows":
        pip_executable += ".exe"

    try:
        print(
            f"Используем {pip_executable} для установки зависимостей из {requirements_file}..."
        )
        subprocess.check_call([pip_executable, "install", "-r", requirements_file])
        print("Все зависимости успешно установлены.")
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при установке зависимостей: {e}")
        sys.exit(1)


def main():
    print("Запуск скрипта...")
    venv_dir = create_virtual_environment()
    install_requirements(venv_dir)
    print("Скрипт выполнен успешно.")


if __name__ == "__main__":
    main()
