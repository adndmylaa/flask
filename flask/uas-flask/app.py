from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

app = Flask(__name__)

# Load model dan data
model = load_model("model_cnn_lstm.h5", compile=False)
df = pd.read_excel("rumah_tangga.xlsx")
df = df[df['Jenis Kelamin'] == 'Perempuan']

# Label encoding
le_daerah = LabelEncoder()
le_umur = LabelEncoder()
df['Daerah_enc'] = le_daerah.fit_transform(df['Daerah'])
df['Umur_enc'] = le_umur.fit_transform(df['Kelompok Umur'])

daerah_list = list(le_daerah.classes_)
umur_list = list(le_umur.classes_)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        daerah_input = request.form["daerah"]
        umur_input = request.form["umur"]
        tahun_input = int(request.form["tahun"])

        # Encode input
        daerah_enc = le_daerah.transform([daerah_input])[0]
        umur_enc = le_umur.transform([umur_input])[0]

        # Filter data sesuai input
        df_filtered = df[
            (df["Daerah_enc"] == daerah_enc) &
            (df["Umur_enc"] == umur_enc) &
            (df["Tahun"] <= tahun_input)
        ].sort_values(by="Tahun")

        # Ambil 3 tahun terakhir
        seq_len = 3
        df_seq = df_filtered.tail(seq_len)

        if len(df_seq) < seq_len:
            return render_template("index.html",
                                   daerah_list=daerah_list,
                                   umur_list=umur_list,
                                   prediction=False,
                                   error="Data tidak cukup untuk prediksi (butuh 3 tahun terakhir).")

        # Drop kolom yang tidak diperlukan
        df_model = df_seq.drop(columns=['No', 'Jenis Kelamin', 'Daerah', 'Kelompok Umur'])

        # Normalisasi
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df_model)

        input_data = scaled.reshape(1, seq_len, scaled.shape[1])

        # Prediksi
        prediction = model.predict(input_data)[0]
        hasil = scaler.inverse_transform([prediction])[0]

        # Persentase
        persen_belum_kawin = hasil[3]
        persen_kawin = hasil[4]
        persen_cerai_hidup = hasil[5]
        persen_cerai_mati = hasil[6]

        # Jumlah populasi (bisa diubah sesuai data nyata)
        total_populasi = 10000

        jumlah_belum_kawin = round((persen_belum_kawin / 100) * total_populasi)
        jumlah_kawin = round((persen_kawin / 100) * total_populasi)
        jumlah_cerai_hidup = round((persen_cerai_hidup / 100) * total_populasi)
        jumlah_cerai_mati = round((persen_cerai_mati / 100) * total_populasi)

        return render_template("index.html",
                               daerah_list=daerah_list,
                               umur_list=umur_list,
                               prediction=True,
                               tahun=tahun_input,
                               persen_belum_kawin=persen_belum_kawin,
                               persen_kawin=persen_kawin,
                               persen_cerai_hidup=persen_cerai_hidup,
                               persen_cerai_mati=persen_cerai_mati,
                               jumlah_belum_kawin=jumlah_belum_kawin,
                               jumlah_kawin=jumlah_kawin,
                               jumlah_cerai_hidup=jumlah_cerai_hidup,
                               jumlah_cerai_mati=jumlah_cerai_mati)

    return render_template("index.html",
                           daerah_list=daerah_list,
                           umur_list=umur_list,
                           prediction=False)

if __name__ == "__main__":
    app.run(debug=True)
