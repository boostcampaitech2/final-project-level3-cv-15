window.onload = function () {
    $("#btn_stream").click(function () {
        document.getElementById("streaming").src = "/video_feed"

        streaming()
    })
}