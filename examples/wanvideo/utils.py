from moviepy.editor import VideoFileClip, concatenate_videoclips


def merge_videos(video_path1, video_path2, output_path):
    """
    拼接两个视频并导出为一个新视频。

    :param video_path1: 第一个视频的文件路径
    :param video_path2: 第二个视频的文件路径
    :param output_path: 输出视频的文件路径
    """
    # 加载视频
    clip1 = VideoFileClip(video_path1)
    clip2 = VideoFileClip(video_path2)

    # 拼接视频
    final_clip = concatenate_videoclips([clip1, clip2])

    # 导出视频
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")


# 示例用法
merge_videos("./mi300_wan14_i2v_720p_lady2_1.mp4", " mi300_wan14_i2v_720p_lady2_2.mp4", "mi300_wan14_i2v_merged_lady2.mp4")
