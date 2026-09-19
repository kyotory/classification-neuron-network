#!/bin/bash

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

TRAIN_DIR="$PROJECT_DIR/data/train"
TEST_DIR="$PROJECT_DIR/data/test"

TRAIN_COUNT=100
TEST_COUNT=30

mkdir -p "$TRAIN_DIR"
mkdir -p "$TEST_DIR"

rm -f "$TRAIN_DIR"/*.txt
rm -f "$TEST_DIR"/*.txt


# ============================================================
# Создание базового шаблона
#
# 0 - круг
# 1 - квадрат
# 2 - треугольник
# ============================================================

create_template()
{
    local class="$1"

    cells=()

    case "$class" in

        0)
            # Круг
            cells=(
                0 0 0 0 0 0 0
                0 0 1 1 1 0 0
                0 1 0 0 0 1 0
                0 1 0 0 0 1 0
                0 1 0 0 0 1 0
                0 0 1 1 1 0 0
                0 0 0 0 0 0 0
            )
            ;;

        1)
            # Квадрат
            cells=(
                0 0 0 0 0 0 0
                0 1 1 1 1 1 0
                0 1 0 0 0 1 0
                0 1 0 0 0 1 0
                0 1 0 0 0 1 0
                0 1 1 1 1 1 0
                0 0 0 0 0 0 0
            )
            ;;

        2)
            # Треугольник
            cells=(
                0 0 0 0 0 0 0
                0 0 0 1 0 0 0
                0 0 1 0 1 0 0
                0 1 0 0 0 1 0
                1 0 0 0 0 0 1
                1 1 1 1 1 1 1
                0 0 0 0 0 0 0
            )
            ;;

        *)
            echo "Unknown class: $class"
            return 1
            ;;
    esac

    return 0
}


# ============================================================
# Проверка, был ли пиксель удалён
# ============================================================

was_removed()
{
    local index="$1"
    local i

    for i in "${removed_pixels[@]}"; do
        if [ "$i" -eq "$index" ]; then
            return 0
        fi
    done

    return 1
}


# ============================================================
# Удаление случайных существующих пикселей
#
# Удаляется от 0 до 3 пикселей.
# ============================================================

remove_random_pixels()
{
    local count=$((RANDOM % 5))
    local removed=0

    removed_pixels=()

    while [ "$removed" -lt "$count" ]; do
        local index=$((RANDOM % 49))

        if [ "${cells[$index]}" -eq 1 ]; then
            cells[$index]=0
            removed_pixels+=("$index")
            removed=$((removed + 1))
        fi
    done
}


# ============================================================
# Добавление случайных пикселей
#
# Добавляется от 0 до 3 пикселей.
# Удалённые пиксели повторно не добавляются.
# ============================================================

add_random_pixels()
{
    local count=$((RANDOM % 5))
    local added=0

    while [ "$added" -lt "$count" ]; do
        local index=$((RANDOM % 49))

        if [ "${cells[$index]}" -eq 0 ] &&
           ! was_removed "$index"; then

            cells[$index]=1
            added=$((added + 1))
        fi
    done
}


# ============================================================
# Запись изображения в файл
# ============================================================

save_image()
{
    local filename="$1"
    local class="$2"

    {
        case "$class" in
            0)
                echo "1 0 0"
                ;;

            1)
                echo "0 1 0"
                ;;

            2)
                echo "0 0 1"
                ;;
        esac

        for row in {0..6}; do
            local line=""

            for col in {0..6}; do
                local index=$((row * 7 + col))

                line="${line}${cells[$index]}"

                if [ "$col" -lt 6 ]; then
                    line="${line} "
                fi
            done

            echo "$line"
        done
    } > "$filename"
}


# ============================================================
# Генерация одного изображения
# ============================================================

generate_image()
{
    local filename="$1"
    local class="$2"

    create_template "$class"

    remove_random_pixels
    add_random_pixels

    save_image "$filename" "$class"
}


# ============================================================
# Генерация обучающей выборки
# ============================================================

echo "Generating training dataset..."

for ((i = 1; i <= TRAIN_COUNT; i++)); do
    class=$(( (i - 1) % 3 ))

    filename=$(printf "%s/sample_%03d.txt" \
        "$TRAIN_DIR" \
        "$i")

    generate_image "$filename" "$class"
done


# ============================================================
# Генерация тестовой выборки
# ============================================================

echo "Generating test dataset..."

for ((i = 1; i <= TEST_COUNT; i++)); do
    class=$(( (i - 1) % 3 ))

    filename=$(printf "%s/sample_%03d.txt" \
        "$TEST_DIR" \
        "$i")

    generate_image "$filename" "$class"
done


echo
echo "Dataset generation completed."
echo
echo "Training samples: $TRAIN_COUNT"
echo "Test samples:     $TEST_COUNT"
echo
echo "Training directory: $TRAIN_DIR"
echo "Test directory:     $TEST_DIR"