from flask import Flask
import threading
import os
import io
import numpy as np
from PIL import Image
import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardMarkup, KeyboardButton
from tensorflow.keras.models import load_model
import datetime
app = Flask(__name__)

@app.route("/")
def home():
    return "GreenCheck Bot is running!"
# --- Ogohlantirishlarni o‘chirish ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# --- Token va bot ---
TOKEN = "7796111756:AAFjm8q8FsMndUdGouAHb5AyxkQQ9Rk9Otw"  # Replace with your actual Telegram bot token
bot = telebot.TeleBot(TOKEN)

# --- Modelni yuklash ---
try:
    model = load_model("best_model.h5")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("Model muvaffaqiyatli yuklandi")
except Exception as e:
    print(f"Modelni yuklashda xatolik: {e}")
    model = None

# --- Klass nomlari va ularning tarjimalari ---
all_class_data = [
    {"id": "apelsin_sitrus_yashil", "uz": "Apelsin Sitrus yashil kasalligi", "en": "Orange Citrus Greening Disease", "ru": "Зеленение цитрусовых апельсина"},
    {"id": "bugdoy_boshoq_qurishi", "uz": "Bug'doy Boshoq qurishi-portlashi", "en": "Wheat Head Blight-Blast", "ru": "Фузариоз колоса пшеницы"},
    {"id": "bugdoy_ildiz_chirishi", "uz": "Bug'doy Ildiz bo'yni va poya pastki qismida chirish", "en": "Wheat Root and Stem Rot", "ru": "Гниль корней и стеблей пшеницы"},
    {"id": "bugdoy_qora_nuqta", "uz": "Bug'doy Qora nuqta kasalligi", "en": "Wheat Black Spot Disease", "ru": "Черная пятнистость пшеницы"},
    {"id": "bugdoy_barg_kuyishi", "uz": "Bug'doy Barg kuyishi", "en": "Wheat Leaf Blight", "ru": "Ожог листьев пшеницы"},
    {"id": "bugdoy_jigarrang_zang", "uz": "Bug'doy Jigarrang zang kasalligi", "en": "Wheat Brown Rust Disease", "ru": "Бурая ржавчина пшеницы"},
    {"id": "bugdoy_sariq_zang", "uz": "Bug'doy Sariq zang", "en": "Wheat Yellow Rust", "ru": "Желтая ржавчина пшеницы"},
    {"id": "bugdoy_soglom", "uz": "Bug'doy Sog'lom", "en": "Wheat Healthy", "ru": "Пшеница Здоровая"},
    {"id": "bugdoy_boshoq_chirishi", "uz": "Bug'doy boshoq chirishi", "en": "Wheat Head Rot", "ru": "Гниль колоса пшеницы"},
    {"id": "bulgarqalampir_bakterial_dog", "uz": "Bulgarqalampir Bakterial dog'", "en": "Bell Pepper Bacterial Spot", "ru": "Бактериальная пятнистость перца"},
    {"id": "bulgarqalampir_soglom", "uz": "Bulgarqalampir Sog'lom", "en": "Bell Pepper Healthy", "ru": "Перец Здоровый"},
    {"id": "chernika_soglom", "uz": "Chernika Sog'lom", "en": "Blueberry Healthy", "ru": "Черника Здоровая"},
    {"id": "kartoshka_erta_kuyish", "uz": "Kartoshka Erta kuyish", "en": "Potato Early Blight", "ru": "Ранняя гниль картофеля"},
    {"id": "kartoshka_kech_kuyish", "uz": "Kartoshka Kech kuyish", "en": "Potato Late Blight", "ru": "Поздняя гниль картофеля"},
    {"id": "kartoshka_soglom", "uz": "Kartoshka Sog'lom", "en": "Potato Healthy", "ru": "Картофель Здоровый"},
    {"id": "makkajoxori_kulrang_dog", "uz": "Makkajoxori Kulrang dog'", "en": "Corn Gray Leaf Spot", "ru": "Серая пятнистость кукурузы"},
    {"id": "makkajoxori_shimoliy_barg_kuyishi", "uz": "Makkajoxori Shimoliy barg kuyishi", "en": "Corn Northern Leaf Blight", "ru": "Северный ожог листьев кукурузы"},
    {"id": "makkajoxori_soglom", "uz": "Makkajoxori Sog'lom", "en": "Corn Healthy", "ru": "Кукуруза Здоровая"},
    {"id": "makkajoxori_zang", "uz": "Makkajoxori Zang", "en": "Corn Rust", "ru": "Ржавчина кукурузы"},
    {"id": "malina_soglom", "uz": "Malina Sog'lom", "en": "Raspberry Healthy", "ru": "Малина Здоровая"},
    {"id": "olcha_soglom", "uz": "Olcha Sog'lom", "en": "Cherry Healthy", "ru": "Вишня Здоровая"},
    {"id": "olcha_un_shudringi", "uz": "Olcha Un shudringi", "en": "Cherry Powdery Mildew", "ru": "Мучнистая роса вишни"},
    {"id": "olma_qotiri", "uz": "Olma Qo'tiri", "en": "Apple Scab", "ru": "Парша яблони"},
    {"id": "olma_qora_chirish", "uz": "Olma Qora chirish", "en": "Apple Black Rot", "ru": "Черная гниль яблони"},
    {"id": "olma_soglom", "uz": "Olma Sog'lom", "en": "Apple Healthy", "ru": "Яблоня Здоровая"},
    {"id": "olma_zang", "uz": "Olma Zang", "en": "Apple Rust", "ru": "Ржавчина яблони"},
    {"id": "paxta_barg_burishma", "uz": "Paxta Barg burishma kasalligi", "en": "Cotton Leaf Curl Disease", "ru": "Болезнь скручивания листьев хлопка"},
    {"id": "paxta_soglom", "uz": "Paxta Sog'lom", "en": "Cotton Healthy", "ru": "Хлопок Здоровый"},
    {"id": "paxta_bakterial_kuyish", "uz": "Paxta bakterial kuyish", "en": "Cotton Bacterial Blight", "ru": "Бактериальный ожог хлопка"},
    {"id": "paxta_fuzarioz_solishi", "uz": "Paxta fuzarioz so'lishi", "en": "Cotton Fusarium Wilt", "ru": "Фузариозное увядание хлопка"},
    {"id": "pomidor_bakterial_dog", "uz": "Pomidor Bakterial dog'", "en": "Tomato Bacterial Spot", "ru": "Бактериальная пятнистость томата"},
    {"id": "pomidor_barg_mogori", "uz": "Pomidor Barg mog'ori", "en": "Tomato Leaf Mold", "ru": "Листовая плесень томата"},
    {"id": "pomidor_erta_kuyish", "uz": "Pomidor Erta kuyish", "en": "Tomato Early Blight", "ru": "Ранняя гниль томата"},
    {"id": "pomidor_halqali_dog", "uz": "Pomidor Halqali dog's", "en": "Tomato Target Spot", "ru": "Кольцевая пятнистость томата"},
    {"id": "pomidor_kech_kuyish", "uz": "Pomidor Kech kuyish", "en": "Tomato Late Blight", "ru": "Поздняя гниль томата"},
    {"id": "pomidor_mozaika_virusi", "uz": "Pomidor Mozaika virusi", "en": "Tomato Mosaic Virus", "ru": "Вирус мозаики томата"},
    {"id": "pomidor_orgimchakkana", "uz": "Pomidor O'rgimchakkana", "en": "Tomato Spider Mites", "ru": "Паутинный клещ томата"},
    {"id": "pomidor_sariq_barg_virusi", "uz": "Pomidor Sariq barg virusi", "en": "Tomato Yellow Leaf Curl Virus", "ru": "Вирус желтой курчаности листьев томата"},
    {"id": "pomidor_septoriya_dogi", "uz": "Pomidor Septoriya dog'i", "en": "Tomato Septoria Leaf Spot", "ru": "Септориозная пятнистость томата"},
    {"id": "pomidor_soglom", "uz": "Pomidor Sog'lom", "en": "Tomato Healthy", "ru": "Томат Здоровый"},
    {"id": "qovoq_un_shudringi", "uz": "Qovoq Un shudringi", "en": "Squash Powdery Mildew", "ru": "Мучнистая роса тыквы"},
    {"id": "qulupnay_barg_kuyishi", "uz": "Qulupnay Barg kuyishi", "en": "Strawberry Leaf Scorch", "ru": "Ожог листьев клубники"},
    {"id": "qulupnay_soglom", "uz": "Qulupnay Sog'lom", "en": "Strawberry Healthy", "ru": "Клубника Здоровая"},
    {"id": "shaftoli_bakterial_dog", "uz": "Shaftoli Bakterial dog'", "en": "Peach Bacterial Spot", "ru": "Бактериальная пятнистость персика"},
    {"id": "shaftoli_soglom", "uz": "Shaftoli Sog'lom", "en": "Peach Healthy", "ru": "Персик Здоровый"},
    {"id": "soya_soglom", "uz": "Soya Sog'lom", "en": "Soybean Healthy", "ru": "Соя Здоровая"},
    {"id": "uzum_barg_kuyishi", "uz": "Uzum Barg kuyishi", "en": "Grape Leaf Blight", "ru": "Ожог листьев винограда"},
    {"id": "uzum_qora_chirish", "uz": "Uzum Qora chirish", "en": "Grape Black Rot", "ru": "Черная гниль винограда"},
    {"id": "uzum_qora_dog", "uz": "Uzum Qora dog'", "en": "Grape Black Spot", "ru": "Черная пятнистость винограда"},
    {"id": "uzum_soglom", "uz": "Uzum Sog'lom", "en": "Grape Healthy", "ru": "Виноград Здоровый"}
]

# class_labels ni model bashoratlari uchun oddiy o'zbekcha ro'yxat sifatida saqlash
class_labels = [item["uz"] for item in all_class_data]

# ID bo'yicha sinf ma'lumotlariga tezkor kirish uchun lug'at
class_data_by_uz_label = {item["uz"]: item for item in all_class_data}

# --- Tavsiyalar ---
# Yangilangan tavsiyalar
recommendations_uz_mapping = {
    "Apelsin Sitrus yashil kasalligi": "Hasharotlarga qarshi Imidacloprid yoki Spirotetramat kabi insektitsidlarni qo‘llang va o‘simlikni muntazam sug‘orib, ozuqa moddalari bilan ta’minlang.",
    "Bug'doy Boshoq qurishi-portlashi": "Propiconazole yoki Tebuconazole fungitsidlarini har 10-14 kunda seping va tuproqni yaxshi drenaj bilan ta’minlang.",
    "Bug'doy Ildiz bo'yni va poya pastki qismida chirish": "Tuproq drenajini yaxshilang va Mancozeb yoki Carbendazim fungitsidlarini qo‘llang.",
    "Bug'doy Qora nuqta kasalligi": "Chlorothalonil yoki Azoxystrobin fungitsidlarini har 7-10 kunda seping va shamollatishni yaxshilang.",
    "Bug'doy Barg kuyishi": "Azoxystrobin yoki Propiconazole fungitsidlarini har 7-10 kunda qo‘llang va o‘simliklar orasidagi masofani optimal saqlang.",
    "Bug'doy Jigarrang zang kasalligi": "Triadimefon yoki Tebuconazole fungitsidlarini har 10-14 kunda seping va o‘simlikni azotli o‘g‘itlar bilan ta’minlang.",
    "Bug'doy Sariq zang": "Mancozeb yoki Propiconazole fungitsidlarini har 10 kunda qo‘llang va namlikni nazorat qiling.",
    "Bug'doy Sog'lom": "Sug‘orish va azotli o‘g‘itlar bilan ta’minlang.",
    "Bug'doy boshoq chirishi": "Propiconazole yoki Azoxystrobin fungitsidlarini har 10-14 kunda seping va tuproq namligini nazorat qiling.",
    "Bulgarqalampir Bakterial dog'": "Streptomycin yoki Copper Hydroxide dorilarini har 7-10 kunda seping va shamollatishni yaxshilang.",
    "Bulgarqalampir Sog'lom": "Namlikni nazorat qiling va o‘g‘it bilan ta’minlang.",
    "Chernika Sog'lom": "Tuproqni mulchalang va namlikni ushlab turing.",
    "Kartoshka Erta kuyish": "Chlorothalonil yoki Mancozeb fungitsidlarini har 7 kunda seping va o‘simliklarni muntazam sug‘oring.",
    "Kartoshka Kech kuyish": "Mancozeb yoki Copper Oxychloride dorilarini har 7-10 kunda qo‘llang va namlikni kamaytiring.",
    "Kartoshka Sog'lom": "Doimiy sug‘orish va kaliyli o‘g‘itlar bilan ta’minlang.",
    "Makkajoxori Kulrang dog'": "Chlorothalonil yoki Azoxystrobin fungitsidlarini har 10 kunda seping va o‘simliklar orasidagi shamollatishni yaxshilang.",
    "Makkajoxori Shimoliy barg kuyishi": "Azoxystrobin yoki Propiconazole fungitsidlarini qo‘llang va kasallik tarqalishini oldini olish uchun izolyatsiya qiling.",
    "Makkajoxori Sog'lom": "Muntazam sug‘oring va o‘g‘it qo‘shing.",
    "Makkajoxori Zang": "Mancozeb yoki Triadimefon dorilarini har 10 kunda seping.",
    "Malina Sog'lom": "Namlikni 60% atrofida saqlang va mulch qo‘llang.",
    "Olcha Sog'lom": "Azotli o‘g‘it qo‘shing va sug‘oring.",
    "Olcha Un shudringi": "Sulfur yoki Myclobutanil dorilarini har 7-10 kunda seping va namlikni nazorat qiling.",
    "Olma Qo'tiri": "Mancozeb yoki Captan fungitsidlarini har 10 kunda seping va daraxtni shamollatishni yaxshilang.",
    "Olma Qora chirish": "Copper Oxychloride yoki Mancozeb dorilarini har 10-14 kunda qo‘llang va o‘simlikni muntazam sug‘oring.",
    "Olma Sog'lom": "Muntazam sug‘oring va kuzatuv olib boring.",
    "Olma Zang": "Triadimefon yoki Propiconazole fungitsidlarini har 10 kunda qo‘llang.",
    "Paxta Barg burishma kasalligi": "Imidacloprid yoki Spirotetramat insektitsidlarini har 10-14 kunda seping va hasharotlarga qarshi kurashing.",
    "Paxta Sog'lom": "Sug‘orish va fosforli o‘g‘itlar bilan ta’minlang.",
    "Paxta bakterial kuyish": "Copper Hydroxide yoki Streptomycin dorilarini har 7-10 kunda seping.",
    "Paxta fuzarioz so'lishi": "Carbendazim yoki Thiophanate-Methyl fungitsidlarini tuproqqa seping va drenajni yaxshilang.",
    "Pomidor Bakterial dog'": "Copper Hydroxide yoki Streptomycin dorilarini har 7-10 kunda seping va shamollatishni yaxshilang.",
    "Pomidor Barg mog'ori": "Azoxystrobin yoki Chlorothalonil fungitsidlarini har 7 kunda qo‘llang va namlikni kamaytiring.",
    "Pomidor Erta kuyish": "Mancozeb yoki Chlorothalonil dorilarini har 10 kunda seping.",
    "Pomidor Halqali dog'": "Chlorothalonil yoki Azoxystrobin fungitsidlarini har 7-10 kunda seping va shamollatishni yaxshilang.",
    "Pomidor Kech kuyish": "Chlorothalonil yoki Mancozeb dorilarini har 7-10 kunda qo‘llang va namlikni nazorat qiling.",
    "Pomidor Mozaika virusi": "Chidamli navlarni tanlang va hasharotlarga qarshi Imidacloprid seping.",
    "Pomidor O'rgimchakkana": "Abamectin yoki Spirotetramat dorilarini har 7-10 kunda qo‘llang.",
    "Pomidor Sariq barg virusi": "Imidacloprid yoki Thiamethoxam insektitsidlarini qo‘llang va hasharotlarga qarshi kurashing.",
    "Pomidor Septoriya dog'i": "Mancozeb yoki Copper Oxychloride dorilarini har 7-10 kunda seping.",
    "Pomidor Sog'lom": "Vaqtida o‘g‘it berib, sug‘orib boring.",
    "Qovoq Un shudringi": "Sulfur yoki Myclobutanil dorilarini har 7-10 kunda seping va namlikni nazorat qiling.",
    "Qulupnay Barg kuyishi": "Copper Oxychloride yoki Mancozeb dorilarini har 7-10 kunda seping va shamollatishni yaxshilang.",
    "Qulupnay Sog'lom": "Namlikni 65% da saqlang.",
    "Shaftoli Bakterial dog'": "Copper Hydroxide yoki Streptomycin dorilarini har 7-10 kunda seping.",
    "Shaftoli Sog'lom": "Fosforli o‘g‘itlar bilan ta’minlang.",
    "Soya Sog'lom": "Azotli o‘g‘it va sug‘orish tizimi bo‘lishi kerak.",
    "Uzum Barg kuyishi": "Azoxystrobin yoki Propiconazole fungitsidlarini har 7-10 kunda seping va shamollatishni yaxshilang.",
    "Uzum Qora chirish": "Captan yoki Mancozeb fungitsidlarini har 10-14 kunda qo‘llang.",
    "Uzum Qora dog'": "Propiconazole yoki Azoxystrobin fungitsidlarini har 10 kunda seping.",
    "Uzum Sog'lom": "O‘g‘it va sug‘orish muvozanatiga rioya qiling."
}
# --- Tavsiyalar va ularning tarjimalari ---
recommendation_translations = {
    "Hasharotlarga qarshi Imidacloprid yoki Spirotetramat kabi insektitsidlarni qo‘llang va o‘simlikni muntazam sug‘orib, ozuqa moddalari bilan ta’minlang.": {
        "en": "Apply insecticides like Imidacloprid or Spirotetramat and ensure regular watering and nutrient supply.",
        "ru": "Применяйте инсектициды, такие как Имидаклоприд или Спиротетрамат, и обеспечьте регулярный полив и питательные вещества."
    },
    "Propiconazole yoki Tebuconazole fungitsidlarini har 10-14 kunda seping va tuproqni yaxshi drenaj bilan ta’minlang.": {
        "en": "Spray Propiconazole or Tebuconazole fungicides every 10-14 days and ensure good soil drainage.",
        "ru": "Опрыскивайте фунгицидами Пропиконазол или Тебуконазол каждые 10-14 дней и обеспечьте хороший дренаж почвы."
    },
    "Tuproq drenajini yaxshilang va Mancozeb yoki Carbendazim fungitsidlarini qo‘llang.": {
        "en": "Improve soil drainage and apply Mancozeb or Carbendazim fungicides.",
        "ru": "Улучшите дренаж почвы и применяйте фунгициды Манкоцеб или Карбендазим."
    },
    "Chlorothalonil yoki Azoxystrobin fungitsidlarini har 7-10 kunda seping va shamollatishni yaxshilang.": {
        "en": "Spray Chlorothalonil or Azoxystrobin fungicides every 7-10 days and improve ventilation.",
        "ru": "Опрыскивайте фунгицидами Хлороталонил или Азоксистробин каждые 7-10 дней и улучшите вентиляцию."
    },
    "Azoxystrobin yoki Propiconazole fungitsidlarini har 7-10 kunda qo‘llang va o‘simliklar orasidagi masofani optimal saqlang.": {
        "en": "Apply Azoxystrobin or Propiconazole fungicides every 7-10 days and maintain optimal spacing between plants.",
        "ru": "Применяйте фунгициды Азоксистробин или Пропиконазол каждые 7-10 дней и поддерживайте оптимальное расстояние между растениями."
    },
    "Triadimefon yoki Tebuconazole fungitsidlarini har 10-14 kunda seping va o‘simlikni azotli o‘g‘itlar bilan ta’minlang.": {
        "en": "Spray Triadimefon or Tebuconazole fungicides every 10-14 days and provide nitrogen fertilizers.",
        "ru": "Опрыскивайте фунгицидами Триадимефон или Тебуконазол каждые 10-14 дней и обеспечьте азотные удобрения."
    },
    "Mancozeb yoki Propiconazole fungitsidlarini har 10 kunda qo‘llang va namlikni nazorat qiling.": {
        "en": "Apply Mancozeb or Propiconazole fungicides every 10 days and control humidity.",
        "ru": "Применяйте фунгициды Манкоцеб или Пропиконазол каждые 10 дней и контролируйте влажность."
    },
    "Sug‘orish va azotli o‘g‘itlar bilan ta’minlang.": {
        "en": "Provide irrigation and nitrogen fertilizers.",
        "ru": "Обеспечьте полив и азотные удобрения."
    },
    "Propiconazole yoki Azoxystrobin fungitsidlarini har 10-14 kunda seping va tuproq namligini nazorat qiling.": {
        "en": "Spray Propiconazole or Azoxystrobin fungicides every 10-14 days and control soil moisture.",
        "ru": "Опрыскивайте фунгицидами Пропиконазол или Азоксистробин каждые 10-14 дней и контролируйте влажность почвы."
    },
    "Streptomycin yoki Copper Hydroxide dorilarini har 7-10 kunda seping va shamollatishni yaxshilang.": {
        "en": "Spray Streptomycin or Copper Hydroxide every 7-10 days and improve ventilation.",
        "ru": "Опрыскивайте Стрептомицином или Гидроксидом меди каждые 7-10 дней и улучшите вентиляцию."
    },
    "Namlikni nazorat qiling va o‘g‘it bilan ta’minlang.": {
        "en": "Control humidity and provide fertilizer.",
        "ru": "Контролируйте влажность и обеспечьте удобрениями."
    },
    "Tuproqni mulchalang va namlikni ushlab turing.": {
        "en": "Mulch the soil and retain moisture.",
        "ru": "Мульчируйте почву и поддерживайте влажность."
    },
    "Chlorothalonil yoki Mancozeb fungitsidlarini har 7 kunda seping va o‘simliklarni muntazam sug‘oring.": {
        "en": "Spray Chlorothalonil or Mancozeb fungicides every 7 days and water plants regularly.",
        "ru": "Опрыскивайте фунгицидами Хлороталонил или Манкоцеб каждые 7 дней и регулярно поливайте растения."
    },
    "Mancozeb yoki Copper Oxychloride dorilarini har 7-10 kunda qo‘llang va namlikni kamaytiring.": {
        "en": "Apply Mancozeb or Copper Oxychloride every 7-10 days and reduce humidity.",
        "ru": "Применяйте Манкоцеб или Оксихлорид меди каждые 7-10 дней и уменьшите влажность."
    },
    "Doimiy sug‘orish va kaliyli o‘g‘itlar bilan ta’minlang.": {
        "en": "Provide regular irrigation and potassium fertilizers.",
        "ru": "Обеспечьте регулярный полив и калийные удобрения."
    },
    "Chlorothalonil yoki Azoxystrobin fungitsidlarini har 10 kunda seping va o‘simliklar orasidagi shamollatishni yaxshilang.": {
        "en": "Spray Chlorothalonil or Azoxystrobin fungicides every 10 days and improve ventilation between plants.",
        "ru": "Опрыскивайте фунгицидами Хлороталонил или Азоксистробин каждые 10 дней и улучшите вентиляцию между растениями."
    },
    "Azoxystrobin yoki Propiconazole fungitsidlarini qo‘llang va kasallik tarqalishini oldini olish uchun izolyatsiya qiling.": {
        "en": "Apply Azoxystrobin or Propiconazole fungicides and isolate to prevent disease spread.",
        "ru": "Применяйте фунгициды Азоксистробин или Пропиконазол и изолируйте, чтобы предотвратить распространение болезни."
    },
    "Muntazam sug‘oring va o‘g‘it qo‘shing.": {
        "en": "Water regularly and add fertilizer.",
        "ru": "Регулярно поливайте и добавляйте удобрения."
    },
    "Mancozeb yoki Triadimefon dorilarini har 10 kunda seping.": {
        "en": "Spray Mancozeb or Triadimefon every 10 days.",
        "ru": "Опрыскивайте Манкоцебом или Триадимефоном каждые 10 дней."
    },
    "Namlikni 60% atrofida saqlang va mulch qo‘llang.": {
        "en": "Maintain humidity around 60% and apply mulch.",
        "ru": "Поддерживайте влажность около 60% и применяйте мульчу."
    },
    "Azotli o‘g‘it qo‘shing va sug‘oring.": {
        "en": "Add nitrogen fertilizer and water.",
        "ru": "Добавьте азотное удобрение и поливайте."
    },
    "Sulfur yoki Myclobutanil dorilarini har 7-10 kunda seping va namlikni nazorat qiling.": {
        "en": "Spray Sulfur or Myclobutanil every 7-10 days and control humidity.",
        "ru": "Опрыскивайте Серой или Миклобутанилом каждые 7-10 дней и контролируйте влажность."
    },
    "Mancozeb yoki Captan fungitsidlarini har 10 kunda seping va daraxtni shamollatishni yaxshilang.": {
        "en": "Spray Mancozeb or Captan fungicides every 10 days and improve tree ventilation.",
        "ru": "Опрыскивайте фунгицидами Манкоцеб или Каптан каждые 10 дней и улучшите вентиляцию дерева."
    },
    "Copper Oxychloride yoki Mancozeb dorilarini har 10-14 kunda qo‘llang va o‘simlikni muntazam sug‘oring.": {
        "en": "Apply Copper Oxychloride or Mancozeb every 10-14 days and water the plant regularly.",
        "ru": "Применяйте Оксихлорид меди или Манкоцеб каждые 10-14 дней и регулярно поливайте растение."
    },
    "Muntazam sug‘oring va kuzatuv olib boring.": {
        "en": "Water regularly and monitor.",
        "ru": "Регулярно поливайте и наблюдайте."
    },
    "Triadimefon yoki Propiconazole fungitsidlarini har 10 kunda qo‘llang.": {
        "en": "Apply Triadimefon or Propiconazole fungicides every 10 days.",
        "ru": "Применяйте фунгициды Триадимефон или Пропиконазол каждые 10 дней."
    },
    "Imidacloprid yoki Spirotetramat insektitsidlarini har 10-14 kunda seping va hasharotlarga qarshi kurashing.": {
        "en": "Spray Imidacloprid or Spirotetramat insecticides every 10-14 days and control pests.",
        "ru": "Опрыскивайте инсектицидами Имидаклоприд или Спиротетрамат каждые 10-14 дней и контролируйте вредителей."
    },
    "Sug‘orish va fosforli o‘g‘itlar bilan ta’minlang.": {
        "en": "Provide irrigation and phosphorus fertilizers.",
        "ru": "Обеспечьте полив и фосфорные удобрения."
    },
    "Copper Hydroxide yoki Streptomycin dorilarini har 7-10 kunda seping.": {
        "en": "Spray Copper Hydroxide or Streptomycin every 7-10 days.",
        "ru": "Опрыскивайте Гидроксидом меди или Стрептомицином каждые 7-10 дней."
    },
    "Carbendazim yoki Thiophanate-Methyl fungitsidlarini tuproqqa seping va drenajni yaxshilang.": {
        "en": "Apply Carbendazim or Thiophanate-Methyl fungicides to the soil and improve drainage.",
        "ru": "Применяйте фунгициды Карбендазим или Тиофанат-Метил в почву и улучшите дренаж."
    },
    "Copper Hydroxide yoki Streptomycin dorilarini har 7-10 kunda seping va shamollatishni yaxshilang.": {
        "en": "Spray Copper Hydroxide or Streptomycin every 7-10 days and improve ventilation.",
        "ru": "Опрыскивайте Гидроксидом меди или Стрептомицином каждые 7-10 дней и улучшите вентиляцию."
    },
    "Azoxystrobin yoki Chlorothalonil fungitsidlarini har 7 kunda qo‘llang va namlikni kamaytiring.": {
        "en": "Apply Azoxystrobin or Chlorothalonil fungicides every 7 days and reduce humidity.",
        "ru": "Применяйте фунгициды Азоксистробин или Хлороталонил каждые 7 дней и уменьшите влажность."
    },
    "Mancozeb yoki Chlorothalonil dorilarini har 10 kunda seping.": {
        "en": "Spray Mancozeb or Chlorothalonil every 10 days.",
        "ru": "Опрыскивайте Манкоцебом или Хлороталонилом каждые 10 дней."
    },
    "Chlorothalonil yoki Azoxystrobin fungitsidlarini har 7-10 kunda seping va shamollatishni yaxshilang.": {
        "en": "Spray Chlorothalonil or Azoxystrobin fungicides every 7-10 days and improve ventilation.",
        "ru": "Опрыскивайте фунгицидами Хлороталонил или Азоксистробин каждые 7-10 дней и улучшите вентиляцию."
    },
    "Chlorothalonil yoki Mancozeb dorilarini har 7-10 kunda qo‘llang va namlikni nazorat qiling.": {
        "en": "Apply Chlorothalonil or Mancozeb every 7-10 days and control humidity.",
        "ru": "Применяйте Хлороталонил или Манкоцеб каждые 7-10 дней и контролируйте влажность."
    },
    "Chidamli navlarni tanlang va hasharotlarga qarshi Imidacloprid seping.": {
        "en": "Choose resistant varieties and spray Imidacloprid against pests.",
        "ru": "Выбирайте устойчивые сорта и опрыскивайте Имидаклопридом против вредителей."
    },
    "Abamectin yoki Spirotetramat dorilarini har 7-10 kunda qo‘llang.": {
        "en": "Apply Abamectin or Spirotetramat every 7-10 days.",
        "ru": "Применяйте Абамектин или Спиротетрамат каждые 7-10 дней."
    },
    "Imidacloprid yoki Thiamethoxam insektitsidlarini qo‘llang va hasharotlarga qarshi kurashing.": {
        "en": "Apply Imidacloprid or Thiamethoxam insecticides and control pests.",
        "ru": "Применяйте инсектициды Имидаклоприд или Тиаметоксам и контролируйте вредителей."
    },
    "Mancozeb yoki Copper Oxychloride dorilarini har 7-10 kunda seping.": {
        "en": "Spray Mancozeb or Copper Oxychloride every 7-10 days.",
        "ru": "Опрыскивайте Манкоцебом или Оксихлоридом меди каждые 7-10 дней."
    },
    "Vaqtida o‘g‘it berib, sug‘orib boring.": {
        "en": "Fertilize and water on time.",
        "ru": "Своевременно удобряйте и поливайте."
    },
    "Sulfur yoki Myclobutanil dorilarini har 7-10 kunda seping va namlikni nazorat qiling.": {
        "en": "Spray Sulfur or Myclobutanil every 7-10 days and control humidity.",
        "ru": "Опрыскивайте Серой или Миклобутанилом каждые 7-10 дней и контролируйте влажность."
    },
    "Copper Oxychloride yoki Mancozeb dorilarini har 7-10 kunda seping va shamollatishni yaxshilang.": {
        "en": "Spray Copper Oxychloride or Mancozeb every 7-10 days and improve ventilation.",
        "ru": "Опрыскивайте Оксихлоридом меди или Манкоцебом каждые 7-10 дней и улучшите вентиляцию."
    },
    "Namlikni 65% da saqlang.": {
        "en": "Maintain humidity at 65%.",
        "ru": "Поддерживайте влажность на уровне 65%."
    },
    "Copper Hydroxide yoki Streptomycin dorilarini har 7-10 kunda seping.": {
        "en": "Spray Copper Hydroxide or Streptomycin every 7-10 days.",
        "ru": "Опрыскивайте Гидроксидом меди или Стрептомицином каждые 7-10 дней."
    },
    "Fosforli o‘g‘itlar bilan ta’minlang.": {
        "en": "Provide phosphorus fertilizers.",
        "ru": "Обеспечьте фосфорные удобрения."
    },
    "Azotli o‘g‘it va sug‘orish tizimi bo‘lishi kerak.": {
        "en": "Nitrogen fertilizer and irrigation system should be in place.",
        "ru": "Должны быть азотные удобрения и система полива."
    },
    "Azoxystrobin yoki Propiconazole fungitsidlarini har 7-10 kunda seping va shamollatishni yaxshilang.": {
        "en": "Spray Azoxystrobin or Propiconazole fungicides every 7-10 days and improve ventilation.",
        "ru": "Опрыскивайте фунгицидами Азоксистробин или Пропиконазол каждые 7-10 дней и улучшите вентиляцию."
    },
    "Captan yoki Mancozeb fungitsidlarini har 10-14 kunda qo‘llang.": {
        "en": "Apply Captan or Mancozeb fungicides every 10-14 days.",
        "ru": "Применяйте фунгициды Каптан или Манкоцеб каждые 10-14 дней."
    },
    "Propiconazole yoki Azoxystrobin fungitsidlarini har 10 kunda seping.": {
        "en": "Spray Propiconazole or Azoxystrobin fungicides every 10 days.",
        "ru": "Опрыскивайте фунгицидами Пропиконазол или Азоксистробин каждые 10 дней."
    },
    "O‘g‘it va sug‘orish muvozanatiga rioya qiling.": {
        "en": "Maintain fertilizer and irrigation balance.",
        "ru": "Соблюдайте баланс удобрений и полива."
    }
}
# --- Ekinlar bo'yicha klassifikatsiya ---
crop_class_mapping = {
    "apelsin": ["Apelsin Sitrus yashil kasalligi"],
    "bug'doy": [
        "Bug'doy Boshoq qurishi-portlashi", "Bug'doy Ildiz bo'yni va poya pastki qismida chirish",
        "Bug'doy Qora nuqta kasalligi", "Bug'doy Barg kuyishi", "Bug'doy Jigarrang zang kasalligi",
        "Bug'doy Sariq zang", "Bug'doy Sog'lom", "Bug'doy boshoq chirishi"
    ],
    "bulgarqalampir": ["Bulgarqalampir Bakterial dog'", "Bulgarqalampir Sog'lom"],
    "chernika": ["Chernika Sog'lom"],
    "kartoshka": ["Kartoshka Erta kuyish", "Kartoshka Kech kuyish", "Kartoshka Sog'lom"],
    "makkajoxori": [
        "Makkajoxori Kulrang dog'", "Makkajoxori Shimoliy barg kuyishi",
        "Makkajoxori Sog'lom", "Makkajoxori Zang"
    ],
    "malina": ["Malina Sog'lom"],
    "olcha": ["Olcha Sog'lom", "Olcha Un shudringi"],
    "olma": ["Olma Qo'tiri", "Olma Qora chirish", "Olma Sog'lom", "Olma Zang"],
    "paxta": [
        "Paxta Barg burishma kasalligi", "Paxta Sog'lom",
        "Paxta bakterial kuyish", "Paxta fuzarioz so'lishi"
    ],
    "pomidor": [
        "Pomidor Bakterial dog'", "Pomidor Barg mog'ori", "Pomidor Erta kuyish",
        "Pomidor Halqali dog'", "Pomidor Kech kuyish", "Pomidor Mozaika virusi",
        "Pomidor O'rgimchakkana", "Pomidor Sariq barg virusi", "Pomidor Septoriya dog'i",
        "Pomidor Sog'lom"
    ],
    "qovoq": ["Qovoq Un shudringi"],
    "qulupnay": ["Qulupnay Barg kuyishi", "Qulupnay Sog'lom"],
    "shaftoli": ["Shaftoli Bakterial dog'", "Shaftoli Sog'lom"],
    "soya": ["Soya Sog'lom"],
    "uzum": ["Uzum Barg kuyishi", "Uzum Qora chirish", "Uzum Qora dog'", "Uzum Sog'lom"]
}

# --- User specific data ---
user_selections = {}  # Stores selected crop for a user
user_languages = {}   # Stores user's preferred language: user_id -> 'uz' | 'en' | 'ru'
user_history = {}     # Stores user's prediction history: user_id -> list of dicts
user_states = {}      # Stores user's current state: user_id -> 'main_menu' | 'crop_selection' | 'history' | 'language_selection'

# --- Translations ---
translations = {
    'uz': {
        'welcome': "Salom! Quyidagi menyudan tanlang.",
        'select_crop': "Iltimos, ekin turini tanlang:",
        'crop_selected': "Siz {crop_name} tanladingiz.",
        'send_photo_prompt': "Endi o'simlik bargi rasmini yuboring. 📸",
        'send_photo_first': "Avval /start orqali ekin turini tanlang.",
        'model_not_loaded': "Model yuklanmagan. Admin bilan bog‘laning.",
        'invalid_image': "Iltimos, {crop_name} o'simlikka mos barg rasmini yuklang. Yuklangan rasm boshqa o'simlikka o'xshaydi yoki sifatsiz.",
        'analysis_results': "📊 <b>{crop_name} tahlil natijalari:</b>\n\n",
        'probable_diseases_heading': "Ehtimoliy kasallanishlar:\n",
        'prediction_summary_detail': "🔹 <b>{class_name}</b>: {confidence:.2f}%\n",
        'overall_conclusion_heading': "\nUmumiy xulosa va tavsiya:\n",
        'overall_conclusion_detail': "🔹 <b>{class_name}</b>\n📝 Tavsiya: {recommendation}\n",
        'error_occurred': "Xatolik yuz berdi: {error_message}. Iltimos, boshqa rasmni sinab ko'ring yoki admin bilan bog'laning.",
        'invalid_choice': "Noto‘g‘ri tanlov!",
        'main_menu_btn_plants': "🌱 O'simliklar",
        'main_menu_btn_language': "🌐 Til",
        'main_menu_btn_history': "📚 Tarix",
        'main_menu_btn_premium': "✨ Premium",
        'select_language': "Iltimos, tilni tanlang:",
        'language_set': "Til muvaffaqiyatli o'rnatildi.",
        'history_empty': "Tarix bo'sh.",
        'history_title': "📚 Tarix:",
        'history_entry': "<b>{timestamp}</b>\nEkin: {crop}\nKasallik: {disease}\nIshonch: {confidence:.2f}%\nTavsiya: {recommendation}\n\n",
        'back_button': "⬅️ Orqaga",
        'recommendation_not_available': "Tavsiya mavjud emas.",
        'premium_under_development': "✨ Premium funksiyalar hozirda texnik ishlar olib borilmoqda. Tez orada ishga tushadi!"
    },
    'en': {
        'welcome': "Hello! Please choose from the menu below.",
        'select_crop': "Please select a crop type:",
        'crop_selected': "You selected {crop_name}.",
        'send_photo_prompt': "Now send an image of the plant leaf. 📸",
        'send_photo_first': "Please select a crop type first using /start.",
        'model_not_loaded': "Model not loaded. Please contact the admin.",
        'invalid_image': "Please upload an image of a leaf matching {crop_name}. The uploaded image appears to belong to another plant or is of low quality.",
        'analysis_results': "📊 <b>{crop_name} analysis results:</b>\n\n",
        'probable_diseases_heading': "Probable diseases:\n",
        'prediction_summary_detail': "🔹 <b>{class_name}</b>: {confidence:.2f}%\n",
        'overall_conclusion_heading': "\nOverall conclusion and recommendation:\n",
        'overall_conclusion_detail': "🔹 <b>{class_name}</b>\n📝 Recommendation: {recommendation}\n",
        'error_occurred': "An error occurred: {error_message}. Please try another image or contact the admin.",
        'invalid_choice': "Invalid choice!",
        'main_menu_btn_plants': "🌱 Plants",
        'main_menu_btn_language': "🌐 Language",
        'main_menu_btn_history': "📚 History",
        'main_menu_btn_premium': "✨ Premium",
        'select_language': "Please select a language:",
        'language_set': "Language set successfully.",
        'history_empty': "History is empty.",
        'history_title': "📚 History:",
        'history_entry': "<b>{timestamp}</b>\nCrop: {crop}\nDisease: {disease}\nConfidence: {confidence:.2f}%\nRecommendation: {recommendation}\n\n",
        'back_button': "⬅️ Back",
        'recommendation_not_available': "Recommendation not available.",
        'premium_under_development': "✨ Premium features are currently under technical development. Coming soon!"
    },
    'ru': {
        'welcome': "Здравствуйте! Пожалуйста, выберите из меню ниже.",
        'select_crop': "Пожалуйста, выберите тип культуры:",
        'crop_selected': "Вы выбрали {crop_name}.",
        'send_photo_prompt': "Теперь отправьте изображение листа растения. 📸",
        'send_photo_first': "Пожалуйста, сначала выберите тип культуры с помощью /start.",
        'model_not_loaded': "Модель не загружена. Пожалуйста, свяжитесь с администратором.",
        'invalid_image': "Пожалуйста, загрузите изображение листа, соответствующее {crop_name}. Загруженное изображение похоже на другое растение или имеет низкое качество.",
        'analysis_results': "📊 <b>Результаты анализа {crop_name}:</b>\n\n",
        'probable_diseases_heading': "Вероятные заболевания:\n",
        'prediction_summary_detail': "🔹 <b>{class_name}</b>: {confidence:.2f}%\n",
        'overall_conclusion_heading': "\nОбщий вывод и рекомендация:\n",
        'overall_conclusion_detail': "🔹 <b>{class_name}</b>\n📝 Рекомендация: {recommendation}\n",
        'error_occurred': "Произошла ошибка: {error_message}. Пожалуйста, попробуйте другое изображение или свяжитесь с администратором.",
        'invalid_choice': "Неверный выбор!",
        'main_menu_btn_plants': "🌱 Растения",
        'main_menu_btn_language': "🌐 Язык",
        'main_menu_btn_history': "📚 История",
        'main_menu_btn_premium': "✨ Премиум",
        'select_language': "Пожалуйста, выберите язык:",
        'language_set': "Язык установлен успешно.",
        'history_empty': "История пуста.",
        'history_title': "📚 История:",
        'history_entry': "<b>{timestamp}</b>\nКультура: {crop}\nБолезнь: {disease}\nУверенность: {confidence:.2f}%\nРекомендация: {recommendation}\n\n",
        'back_button': "⬅️ Назад",
        'recommendation_not_available': "Рекомендация недоступна.",
        'premium_under_development': "✨ Премиум-функции в настоящее время находятся в технической разработке. Скоро будет доступно!"
    }
}

def get_text(user_id, key, category=None, **kwargs):
    lang = user_languages.get(user_id, 'uz')
    if category == 'class_label':
        class_data = class_data_by_uz_label.get(key)
        return class_data.get(lang, key) if class_data else key
    elif category == 'recommendation':
        rec_data = recommendation_translations.get(key)
        return rec_data.get(lang, key) if rec_data else key
    elif category == 'crop_name':
        crop_name_translations = {
            "apelsin": {"uz": "Apelsin", "en": "Orange", "ru": "Апельсин"},
            "bug'doy": {"uz": "Bug'doy", "en": "Wheat", "ru": "Пшеница"},
            "bulgarqalampir": {"uz": "Bulg'orqalampir", "en": "Bell Pepper", "ru": "Болгарский перец"},
            "chernika": {"uz": "Chernika", "en": "Blueberry", "ru": "Черника"},
            "kartoshka": {"uz": "Kartoshka", "en": "Potato", "ru": "Картофель"},
            "makkajoxori": {"uz": "Makkajo'xori", "en": "Corn", "ru": "Кукуруза"},
            "malina": {"uz": "Malina", "en": "Raspberry", "ru": "Малина"},
            "olcha": {"uz": "Olcha", "en": "Cherry", "ru": "Вишня"},
            "olma": {"uz": "Olma", "en": "Apple", "ru": "Яблоня"},
            "paxta": {"uz": "Paxta", "en": "Cotton", "ru": "Хлопок"},
            "pomidor": {"uz": "Pomidor", "en": "Tomato", "ru": "Помидор"},
            "qovoq": {"uz": "Qovoq", "en": "Squash", "ru": "Тыква"},
            "qulupnay": {"uz": "Qulupnay", "en": "Strawberry", "ru": "Клубника"},
            "shaftoli": {"uz": "Shaftoli", "en": "Peach", "ru": "Персик"},
            "soya": {"uz": "Soya", "en": "Soybean", "ru": "Соя"},
            "uzum": {"uz": "Uzum", "en": "Grape", "ru": "Виноград"},
        }
        return crop_name_translations.get(key, {}).get(lang, key)
    else:
        return translations[lang].get(key, f"Translation missing for {key} in {lang}").format(**kwargs)

# --- Doimiy footer tugmalar paneli (ReplyKeyboardMarkup) ---
def get_main_menu_reply_keyboard(user_id):
    keyboard = ReplyKeyboardMarkup(row_width=2, resize_keyboard=True, one_time_keyboard=False)
    keyboard.add(
        KeyboardButton(get_text(user_id, 'main_menu_btn_plants')),
        KeyboardButton(get_text(user_id, 'main_menu_btn_language'))
    )
    keyboard.add(
        KeyboardButton(get_text(user_id, 'main_menu_btn_history')),
        KeyboardButton(get_text(user_id, 'main_menu_btn_premium'))
    )
    return keyboard

# --- Inline tugmalar ---
def get_crop_keyboard(user_id):
    keyboard = InlineKeyboardMarkup(row_width=2)
    for crop_key in crop_class_mapping.keys():
        translated_crop_name = get_text(user_id, crop_key, category='crop_name')
        keyboard.add(InlineKeyboardButton(translated_crop_name, callback_data=f"crop_{crop_key}"))
    keyboard.add(InlineKeyboardButton(get_text(user_id, 'back_button'), callback_data="back_to_main"))
    return keyboard

def get_language_keyboard(user_id):
    keyboard = InlineKeyboardMarkup(row_width=2)
    keyboard.add(
        InlineKeyboardButton("🇺🇿 O'zbek", callback_data="lang_uz"),
        InlineKeyboardButton("🇬🇧 English", callback_data="lang_en")
    )
    keyboard.add(
        InlineKeyboardButton("🇷🇺 Русский", callback_data="lang_ru"),
        InlineKeyboardButton(get_text(user_id, 'back_button'), callback_data="back_to_main")
    )
    return keyboard

def get_history_keyboard(user_id):
    keyboard = InlineKeyboardMarkup(row_width=1)
    keyboard.add(InlineKeyboardButton(get_text(user_id, 'back_button'), callback_data="back_to_main"))
    return keyboard

# --- Rasmni tayyorlash ---
def prepare_image(img_data):
    try:
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        w, h = img.size
        min_dim = min(w, h)
        left = (w - min_dim) / 2
        top = (h - min_dim) / 2
        right = (w + min_dim) / 2
        bottom = (h + min_dim) / 2
        img = img.crop((left, top, right, bottom))
        img = img.resize((224, 224))
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        raise Exception(f"Rasmni tayyorlashda xatolik: {str(e)}")
# --- O'simlikka moslikni tekshirish (yangilangan usul) ---
def is_valid_plant_image(predictions, crop_classes_uz, min_overall_confidence=0.10, min_relevant_confidence_for_match=0.05):
    """
    Checks if the image is likely a plant leaf relevant to the selected crop.
    
    Args:
        predictions (np.array): Raw prediction probabilities from the model for all classes.
        crop_classes_uz (list): List of class labels (UZ) relevant to the selected crop.
        min_overall_confidence (float): Minimal ishonchlilik chegarasi. Agar modelning
                                        eng yuqori bashorati ushbu chegaradan past bo'lsa,
                                        rasm model uchun notanish yoki sifatsiz deb hisoblanadi.
        min_relevant_confidence_for_match (float): Tanlangan ekin sinflari ichidagi eng yuqori bashorat
                                         uchun minimal ishonchlilik chegarasi. Agar bu chegaradan
                                         past bo'lsa, rasm tanlangan ekinga mos kelmaydi deb hisoblanadi.
    Returns:
        bool: True if the image is considered valid for the selected crop, False otherwise.
    """
    if not predictions.size:
        return False

    # 1. Umumiy eng yuqori ishonchlilikni tekshirish (rasm umuman o'simlikka o'xshaydimi)
    top_confidence_overall = np.max(predictions)
    if top_confidence_overall < min_overall_confidence:
        return False  # Rasm sifatsiz yoki o'simlik emas

    # 2. Tanlangan o'simlik sinflari bo'yicha ishonchlilikni tekshirish
    relevant_indices = []
    for cls in crop_classes_uz:
        try:
            idx = class_labels.index(cls)
            relevant_indices.append(idx)
        except ValueError:
            continue

    if not relevant_indices:
        return False  # Hech qanday mos sinf topilmadi

    # Tanlangan o'simlik sinflari bo'yicha maksimal ishonchlilikni aniqlash
    max_confidence_in_crop = max([predictions[idx] for idx in relevant_indices if idx < len(predictions)] or [0.0])
    if max_confidence_in_crop < min_relevant_confidence_for_match:
        return False  # Tanlangan o'simlikka mos ishonchlilik yetarli emas

    return True

# --- /start komandasi ---
@bot.message_handler(commands=['start'])
def start(message):
    user_id = message.from_user.id
    user_languages.setdefault(user_id, 'uz')
    user_states[user_id] = 'main_menu'
    bot.send_message(message.chat.id, get_text(user_id, 'welcome'), reply_markup=get_main_menu_reply_keyboard(user_id))

# --- Matn xabarlarini qabul qilish ---
@bot.message_handler(func=lambda message: message.text in [
    translations['uz']['main_menu_btn_plants'], translations['en']['main_menu_btn_plants'], translations['ru']['main_menu_btn_plants'],
    translations['uz']['main_menu_btn_language'], translations['en']['main_menu_btn_language'], translations['ru']['main_menu_btn_language'],
    translations['uz']['main_menu_btn_history'], translations['en']['main_menu_btn_history'], translations['ru']['main_menu_btn_history'],
    translations['uz']['main_menu_btn_premium'], translations['en']['main_menu_btn_premium'], translations['ru']['main_menu_btn_premium']])
def handle_main_menu_text_buttons(message):
    user_id = message.from_user.id
    chat_id = message.chat.id
    text = message.text
    if text == get_text(user_id, 'main_menu_btn_plants'):
        user_states[user_id] = 'crop_selection'
        bot.send_message(chat_id, get_text(user_id, 'select_crop'), reply_markup=get_crop_keyboard(user_id))
    elif text == get_text(user_id, 'main_menu_btn_language'):
        user_states[user_id] = 'language_selection'
        bot.send_message(chat_id, get_text(user_id, 'select_language'), reply_markup=get_language_keyboard(user_id))
    elif text == get_text(user_id, 'main_menu_btn_history'):
        user_states[user_id] = 'history'
        history_entries = user_history.get(user_id, [])
        if not history_entries:
            history_message = get_text(user_id, 'history_empty')
        else:
            history_message = get_text(user_id, 'history_title') + "\n\n"
            for entry in history_entries:
                translated_crop = get_text(user_id, entry['crop_key'], category='crop_name')
                translated_disease = get_text(user_id, entry['disease_uz'], category='class_label')
                translated_recommendation = get_text(user_id, entry['recommendation_uz'], category='recommendation')
                history_message += get_text(user_id, 'history_entry',
                                           timestamp=entry['timestamp'],
                                           crop=translated_crop,
                                           disease=translated_disease,
                                           confidence=float(entry['confidence']),
                                           recommendation=translated_recommendation)
        bot.send_message(chat_id, history_message, reply_markup=get_history_keyboard(user_id), parse_mode="HTML")
    elif text == get_text(user_id, 'main_menu_btn_premium'):
        bot.send_message(chat_id, get_text(user_id, 'premium_under_development'), reply_markup=get_main_menu_reply_keyboard(user_id))

# --- Tugma bosish ---
@bot.callback_query_handler(func=lambda call: True)
def callback_query(call):
    user_id = call.from_user.id
    data = call.data
    chat_id = call.message.chat.id
    message_id = call.message.message_id
    if data.startswith("lang_"):
        lang_code = data.split("_")[1]
        user_languages[user_id] = lang_code
        bot.answer_callback_query(call.id, text=get_text(user_id, 'language_set'))
        user_states[user_id] = 'main_menu'
        bot.edit_message_reply_markup(chat_id=chat_id, message_id=message_id, reply_markup=None)
        bot.send_message(chat_id, get_text(user_id, 'welcome'), reply_markup=get_main_menu_reply_keyboard(user_id))
    elif data.startswith("crop_"):
        selected_crop_key = data.split("_")[1]
        if selected_crop_key not in crop_class_mapping:
            bot.answer_callback_query(call.id, text=get_text(user_id, 'invalid_choice'))
            return
        user_selections[user_id] = selected_crop_key
        bot.answer_callback_query(call.id)
        bot.edit_message_reply_markup(chat_id=chat_id, message_id=message_id, reply_markup=None)
        translated_crop_name = get_text(user_id, selected_crop_key, category='crop_name')
        bot.send_message(chat_id, get_text(user_id, 'crop_selected', crop_name=translated_crop_name))
        bot.send_message(chat_id, get_text(user_id, 'send_photo_prompt'), reply_markup=get_main_menu_reply_keyboard(user_id))
    elif data == "back_to_main":
        user_states[user_id] = 'main_menu'
        bot.edit_message_reply_markup(chat_id=chat_id, message_id=message_id, reply_markup=None)
        bot.send_message(chat_id, get_text(user_id, 'welcome'), reply_markup=get_main_menu_reply_keyboard(user_id))
        bot.answer_callback_query(call.id)
    else:
        bot.answer_callback_query(call.id, text=get_text(user_id, 'invalid_choice'))

# --- Rasmni qabul qilish va tahlil qilish ---
@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    user_id = message.from_user.id
    chat_id = message.chat.id

    if user_id not in user_selections:
        bot.reply_to(message, get_text(user_id, 'send_photo_first'), reply_markup=get_main_menu_reply_keyboard(user_id))
        return

    if not model:
        bot.reply_to(message, get_text(user_id, 'model_not_loaded'), reply_markup=get_main_menu_reply_keyboard(user_id))
        return

    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        img_array = prepare_image(downloaded_file)

        raw_predictions = model.predict(img_array)[0]  # Original bashoratlarni saqlash
        selected_crop_key = user_selections[user_id]
        crop_classes_uz = crop_class_mapping[selected_crop_key]
        translated_crop_name = get_text(user_id, selected_crop_key, category='crop_name')

        # 1-qadam: Rasm tanlangan o'simlikka mos keladimi tekshirish
        if not is_valid_plant_image(raw_predictions, crop_classes_uz, min_overall_confidence=0.10, min_relevant_confidence_for_match=0.05):
            bot.reply_to(message, get_text(user_id, 'invalid_image', crop_name=translated_crop_name), reply_markup=get_main_menu_reply_keyboard(user_id))
            del user_selections[user_id]
            return

        # 2-qadam: Tanlangan o'simlik uchun bashoratlarni filtrlash va qayta normallashtirish
        relevant_class_indices = []
        for cls in crop_classes_uz:
            try:
                idx = class_labels.index(cls)
                relevant_class_indices.append(idx)
            except ValueError:
                continue

        if not relevant_class_indices:
            bot.reply_to(message, get_text(user_id, 'invalid_image', crop_name=translated_crop_name), reply_markup=get_main_menu_reply_keyboard(user_id))
            del user_selections[user_id]
            return

        # Filtrlangan bashoratlarni yaratish
        filtered_predictions = np.zeros_like(raw_predictions)
        for idx in relevant_class_indices:
            if idx < len(raw_predictions):
                filtered_predictions[idx] = raw_predictions[idx]

        # Ehtimolliklarni qayta normallashtirish
        total_filtered_sum = np.sum(filtered_predictions)
        if total_filtered_sum > 0:
            filtered_predictions = filtered_predictions / total_filtered_sum
        else:
            bot.reply_to(message, get_text(user_id, 'invalid_image', crop_name=translated_crop_name), reply_markup=get_main_menu_reply_keyboard(user_id))
            del user_selections[user_id]
            return

        # 3-qadam: Eng yuqori 3 ta bashoratni aniqlash
        relevant_preds_with_conf = [(idx, filtered_predictions[idx]) for idx in relevant_class_indices if idx < len(filtered_predictions)]
        relevant_preds_with_conf.sort(key=lambda x: x[1], reverse=True)
        top3_indices_filtered = [item[0] for item in relevant_preds_with_conf[:3]]

        # 4-qadam: Eng yuqori bashoratning ishonchliligini tekshirish
        if not top3_indices_filtered or (filtered_predictions[top3_indices_filtered[0]] * 100) < 20:  # Chegarani 20% ga tushirdim
            bot.reply_to(message, get_text(user_id, 'invalid_image', crop_name=translated_crop_name), reply_markup=get_main_menu_reply_keyboard(user_id))
            del user_selections[user_id]
            return

        # Natija xabarini shakllantirish
        result_message = get_text(user_id, 'analysis_results', crop_name=translated_crop_name)
        result_message += get_text(user_id, 'probable_diseases_heading')

        for idx in top3_indices_filtered:
            label_uz = class_labels[idx]
            translated_label = get_text(user_id, label_uz, category='class_label')
            result_message += get_text(user_id, 'prediction_summary_detail',
                                      class_name=translated_label,
                                      confidence=float(filtered_predictions[idx] * 100))

        # Umumiy xulosa va tavsiya
        top_pred_idx = top3_indices_filtered[0]
        top_label_uz = class_labels[top_pred_idx]
        top_recommendation_uz = recommendations_uz_mapping.get(top_label_uz, get_text(user_id, 'recommendation_not_available'))
        
        translated_top_label = get_text(user_id, top_label_uz, category='class_label')
        translated_top_recommendation = get_text(user_id, top_recommendation_uz, category='recommendation')
        
        result_message += get_text(user_id, 'overall_conclusion_heading')
        result_message += get_text(user_id, 'overall_conclusion_detail',
                                  class_name=translated_top_label,
                                  recommendation=translated_top_recommendation)

        # Tarixga qo'shish
        user_history.setdefault(user_id, []).append({
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "crop_key": selected_crop_key,
            "disease_uz": top_label_uz,
            "confidence": float(filtered_predictions[top_pred_idx] * 100),
            "recommendation_uz": top_recommendation_uz
        })

        bot.send_message(chat_id, result_message, parse_mode="HTML", reply_markup=get_main_menu_reply_keyboard(user_id))
        del user_selections[user_id]

    except Exception as e:
        bot.reply_to(message, get_text(user_id, 'error_occurred', error_message=str(e)), reply_markup=get_main_menu_reply_keyboard(user_id))

# --- Botni ishga tushurish ---
def run_bot():
    bot.infinity_polling(timeout=60, long_polling_timeout=30)


if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    print("Bot ishga tushdi...")

    # Botni alohida thread'da ishga tushiramiz
    threading.Thread(target=run_bot, daemon=True).start()

    # Flask serverni ishga tushiramiz (Render PORT'ni ushlab turish uchun muhim)
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
