function streaming(file_path=""){
    var set_up = false
    var traffic = traffic_control()

    url = "http://34.64.139.254:8000/video_feed"
    //url = "http://localhost:8000/video_feed"
    if (file_path != "") url += "?file_path="+file_path

    var source = new EventSource(url);

    source.onmessage = function(event) {
        var data = JSON.parse(event.data)

        if (!set_up) {
            traffic.traffic_start()
            set_up=true
        }
        if (data.ret === false){
            document.getElementById("log-1").innerHTML = "횡단보도 : 미감지";
            source.close()
        }

        document.getElementById("streaming").src = "data:image/jpeg;base64, " + data.img;

        document.getElementById("log-1").innerHTML = "횡단보도 : 감지";
        document.getElementById("log-2").innerHTML = '감지된 박스<br>'+JSON.stringify(data.inform.log.all_box);
        document.getElementById("log-3").innerHTML = '횡단보도 내 박스<br>'+JSON.stringify(data.inform.log.roi_box);

        if (data.inform.traffic.keep_green === true)
            traffic.set_keep_green(true)
        else
            traffic.set_keep_green(false)

        if (data.inform.traffic.keep_red === true)
            traffic.set_keep_red(true)
        else
            traffic.set_keep_red(false)

    };
}