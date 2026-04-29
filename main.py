import os
import tempfile
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram.filters import CommandStart
import asyncio
from dotenv import load_dotenv

from qwenocr import process_document

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    print("BOT_TOKEN не задан. Создайте файл .env с токеном бота.")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()


@dp.message(CommandStart())
async def start_handler(message: Message):
    await message.answer("Отправь файл (PDF или изображение)")


@dp.message(lambda message: message.document or message.photo)
async def file_handler(message: Message):
    file = None

    if message.document:
        file = await bot.get_file(message.document.file_id)
        file_name = message.document.file_name
    elif message.photo:
        file = await bot.get_file(message.photo[-1].file_id)
        file_name = "image.jpg"

    file_path = file.file_path

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, file_name)
        output_path = os.path.join(tmpdir, "result.txt")

        await bot.download_file(file_path, input_path)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, process_document, input_path, output_path)

        with open(output_path, "rb") as f:
            await message.answer_document(f)

async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())