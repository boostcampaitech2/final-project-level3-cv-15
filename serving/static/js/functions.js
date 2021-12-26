window.onload = function () {
    $("#btn_stream").click(function () {
        streaming()
    })

    $("#form-id").submit(function(e) {
        e.preventDefault(); // avoid to execute the actual submit of the form.
        var form = $(this)[0];
        var formData = new FormData(form);

        $.ajax({
            type: "POST",
            enctype: 'multipart/form-data',
            url: 'video_upload',
            data: formData, // serializes the form's elements.
            processData: false,
            contentType: false,
            success: function(data)
            {
               streaming(data)
            },
            error: function (e) {
                console.log("에러:", e)
            }
         });
    });
    
}