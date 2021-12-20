//window.onload = function () {
function traffic() {
    var is_red = true

    //var red_time = parseInt(document.getElementById("red_time").innerText)
    var red_time = parseInt($("#red_time").val())
    var green_time = parseInt($("#green_time").val())
    var time = red_time

    var keep_green = false

    function clock() {
        if (time < 0 && is_red) set_green()
        else if (time < 0) set_red()

        if (!(keep_green && !is_red && time<1)) show_left_time(time--)
    }

    function set_red() {
        // set to RED
        document.getElementById("traffic").src = "/static/src/img/red.jpg"

        time = red_time
        is_red = true
    }

    function set_green() {
        // set to GREEN
        document.getElementById("traffic").src = "/static/src/img/green.jpg"

        time = green_time
        is_red = false
    }

    function show_left_time(time) {
        document.getElementById("left_time").innerHTML = time;
    }

    // 강제 신호 변화 버튼튼
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

    // 1초마다 반복
    clock();
    setInterval(clock, 1000);
}