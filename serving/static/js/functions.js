window.onload = function () {
    $("#btn_stream").click(function () {
        document.getElementById("streaming").src = "/video_feed"
        document.getElementById("traffic").src = "/static/src/img/red.jpg"

        streaming()
    })
}