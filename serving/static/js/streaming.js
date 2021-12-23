function streaming(){
    var traffic = traffic_control()

    var source = new EventSource("http://localhost:8000/video_feed");
    source.onmessage = function(event) {
        console.log(event.data)
        var data = JSON.parse(event.data)

        document.getElementById("logs").innerHTML = "box num : "+data.inform + "<br>";
        document.getElementById("streaming").src = "data:image/jpeg;base64, " + data.img;

        if (data.keep_green === 'true')
            traffic.set_keep_green(true)
        else
            traffic.set_keep_green(false)

        if (data.not_keep_red === 'true')
            traffic.set_keep_red(false)
        else
            traffic.set_keep_red(true)
    };
}

/*
window.onload = function () {
    $.ajax({
        type: 'GET',
        url: '/video_feed',
        error: function (xhr, status, error) {
            console.log(error);
        },
        success: function (json) {
            console.log(json)
        },
    });
}

 */

/*
//window.onload = function () {
$(document).ready(function(){
    var lastResponseLength = false;
    var cnt = 0;
    var idx_list = [ 220518, 220965+220518]
    var idx_list2 = [ 220557,
                     220557+221004,
                     221833 ,
                     224548 ,
                     222647]
    var ajaxRequest = $.ajax({
        type: 'get',
        url: '/video_feed',
        processData: false,
        xhrFields: {
            // Getting on progress streaming response
            onprogress: function (e) {
                var progressResponse;
                var response = e.currentTarget.response;
                if (lastResponseLength === false) {
                    progressResponse = response;
                    lastResponseLength = response.length;
                } else {
                    progressResponse = response.substring(lastResponseLength);
                    lastResponseLength = response.length;
                }
                if (cnt < 1) {

                    if (lastResponseLength > idx_list[cnt]){
                        console.log(lastResponseLength)
                        console.log(idx_list[cnt])
                        progressResponse = response.substring(idx_list[cnt++])
                        document.getElementById("streaming").src = "--frame Content-Type: image/jpeg " + progressResponse
                    }
                }
                //var parsedResponse = JSON.parse(progressResponse);
                //$('#fullResponse').text(parsedResponse.message);
                //$('.progress-bar').css('width', parsedResponse.progress + '%');
            }
        }
    })
})
*/
/*
$(document).ready(function(){
    cnt = 0
    $.stream({
        url:'/video_feed1',
        type: 'GET'
    }).progress(function(response, textStatus, jqXHR){
        console.log('Received: '+response);
    })
    .done(function(packet){
        console.log('And we are done!');
    });
});
*/
