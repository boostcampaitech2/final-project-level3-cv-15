//window.onload = function () {
var traffic_control = function(){
    var is_red = false

    //var red_time = parseInt(document.getElementById("red_time").innerText)
    var red_time = parseInt($("#red_time").val())
    var green_time = parseInt($("#green_time").val())
    var time = red_time

    var keep_green = false
    var keep_red = false

    // 강제 신호 변화 버튼
    $("#set_red").click(function(){
        set_red()
        show_left_time(time)
    })
    $("#set_green").click(function(){
        set_green()
        show_left_time(time)
    })

    // 신호 시간 변경
    $(".time_input").on("propertychange change keyup paste input", function() {
        red_time = parseInt($("#red_time").val())
        green_time = parseInt($("#green_time").val())
    });

    function clock() {
        if (time < 0 && is_red && !keep_red) set_green()
        else if (time < 0) {
            set_red(is_red)
        }
        
        if (!(keep_green && !is_red && time<1)) {
            show_left_time(time--)
        }
        else {
            document.getElementById("keep_green").innerHTML = '신호 유지';
        }
    }

    function show_left_time(time) {
        document.getElementById("left_time").innerHTML = time;
    }

    function set_green(){
        $('.traffic-green').css({
            "background-color":"rgb(53, 218, 103)"
        })
        $('.traffic-red').css({
            "background-color":"gray"
        })
        $('.traffic-text').css({
            "background-color":"rgb(53, 218, 103)"
        })
        document.getElementById("keep_green").innerHTML = '';

        time = green_time
        is_red = false
    }

    function set_red() {
        $('.traffic-red').css({
            "background-color":"rgb(223, 58, 58)"
        })
        $('.traffic-green').css({
            "background-color":"gray"
        })
        $('.traffic-text').css({
            "background-color":"rgb(223, 58, 58)"
        })

        if(is_red) document.getElementById("keep_green").innerHTML = '스킵됨';
        else document.getElementById("keep_green").innerHTML = '';
        time = red_time
        is_red = true
    }

    return {
        set_keep_green: function (bool) {
            keep_green = bool
        },

        set_keep_red: function (bool) {
            keep_red = bool
        },

        traffic_start: function (){
            set_red()
            // 1초마다 반복
            clock();
            setInterval(clock, 1000);
        }
    }
}