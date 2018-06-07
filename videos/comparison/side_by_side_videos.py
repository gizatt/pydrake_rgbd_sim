import os

models = ["cnn",
          "cnn_singlescene",
          "model_wnoise",
          "model_shadowing_and_normals"]
video_names = ["rgb_plus_mask",
               "rgb_plus_masked_depth"]
N = len(models)
for i in range(N):
    for j in range(i+1, N):
        model_1 = models[i]
        model_2 = models[j]

        subdir_name = "zzz_%s_vs_%s" % (model_1, model_2)
        os.system("mkdir -p " + subdir_name)
        for video_name in video_names:
            os.system(
                "avconv -y -i {model_1}/{video_name}.mp4 -i "
                "{model_2}/{video_name}.mp4 -filter_complex "
                "hstack zzz_{model_1}_vs_{model_2}/{video_name}.mp4"
                .format(model_1=model_1, model_2=model_2,
                        video_name=video_name))
