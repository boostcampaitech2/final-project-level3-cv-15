window.onload = function () {
    $("#btn_stream").click(function () {
        streaming()
    })

    $("#btn_submit").click(function () {
        // alert("hihiiihihihihih")
        // file_list = document.getElementById("file_list").value
        // console.log(file_list)
        // console.log("???")
        alert("btn_submit")
        // streaming_upload(file_list)
    })

    $("#form-id").submit(function(e) {
        e.preventDefault(); // avoid to execute the actual submit of the form.
        var form = $(this);
        // var formData = new FormData(form);
        var url = form.attr('action');
        console.log(form)
        console.log(form.serialize())
        console.log(url)
        $.ajax({
               type: "POST",
               url: url,
               data: form.serialize(), // serializes the form's elements.
               success: function(data)
               {
                   alert(data); // show response from the php script.
               }
             });
    });
    
}