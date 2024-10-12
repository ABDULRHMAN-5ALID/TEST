import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import configparser

# إعداد المسار إلى البيانات
base_dir = r'C:\Users\abmg2\Desktop\data_1\archive(1)\database\training'

# قائمة لتخزين البيانات والتسميات
images = []
labels = []

# دالة لتحديد التصنيف بناءً على مجموعة المريض
def get_patient_label(group):
    group_to_label = {
        'NOR': 0,  # Normal
        'MINF': 1,  # Myocardial Infarction
        'DCM': 2,   # Dilated Cardiomyopathy
        'HCM': 3,   # Hypertrophic Cardiomyopathy
        'RV': 4    # Arrhythmogenic Right Ventricular Cardiomyopathy
    }
    return group_to_label.get(group, -1)  # تعيين التصنيف بناءً على المجموعة

def read_patient_info(file_path):
    info = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # تجاهل الأسطر الفارغة
                key, value = line.split(':', 1)
                info[key.strip()] = value.strip()
    return info

# قراءة بيانات التدريب
for patient_folder in os.listdir(base_dir):
    patient_path = os.path.join(base_dir, patient_folder)

    if os.path.isdir(patient_path):
        cfg_file = os.path.join(patient_path, 'Info.cfg')
        if os.path.exists(cfg_file):
            print(f"التحقق من ملف التكوين: {cfg_file}")
            patient_info = read_patient_info(cfg_file)
            patient_group = patient_info.get('Group')
            patient_label = get_patient_label(patient_group)

            # تحقق من صحة التسمية
            if patient_label == -1:
                print(f"تسمية غير صالحة للمريض: {patient_folder}. سيتم تخطيه.")
                continue  # تخطي هذا المريض

            # قراءة ملفات .nii (نحن نركز على الملفات ذات الصلة)
            nii_files = [f for f in os.listdir(patient_path) if f.endswith('.nii')]
            for nii_file in nii_files:
                if 'frame01' in nii_file:  # قد ترغب في استخدام بعض الملفات فقط في البداية
                    nii_path = os.path.join(patient_path, nii_file)
                    print(f"تحميل ملف NII: {nii_path}")
                    img = nib.load(nii_path)
                    data = img.get_fdata()

                    # إضافة البيانات (نقوم بتكديس المحاور الثلاثية لإنشاء صورة 2D)
                    for i in range(data.shape[2]):  # المحور الثالث هو عدد الشرائح
                        img_slice = data[:, :, i]
                        img_resized = resize(img_slice, (200, 600), mode='reflect')  # تغيير الحجم للصورة
                        images.append(img_resized)
                        labels.append(patient_label)  # تعيين التسمية بناءً على المريض

        else:
            print(f"ملف التكوين {cfg_file} غير موجود.")

# تحويل القوائم إلى numpy arrays
images = np.array(images)
labels = np.array(labels)

# طباعة معلومات حول البيانات للتأكد من تحميلها بشكل صحيح
print(f"عدد الصور: {len(images)}")
print(f"عدد التسميات: {len(labels)}")
print(f"حجم الصور: {images.shape if len(images) > 0 else 'لا توجد بيانات'}")

# تطبيع البيانات
if len(images) > 0:
    images = images / np.max(images)

    # تقسيم البيانات إلى مجموعة تدريب واختبار
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # إضافة محور القناة (Channel) إذا لزم الأمر
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # إنشاء نموذج CNN
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(5, activation='softmax')  # 5 فئات
    ])

    # تجميع النموذج
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # تدريب النموذج
    history = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

    # تقييم النموذج على مجموعة الاختبار
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc}")

else:
    print("لا توجد بيانات لتحميلها.")
