document.addEventListener('drop', function (e) {
	e.preventDefault()
}, false)

document.addEventListener('dragover', function (e) {
    e.preventDefault()
}, false)

function openFile(){
	event.stopPropagation();
	document.getElementById("i1").style.display='block';
	document.getElementById("i2").style.display='none';
	document.getElementById("i3").style.display='none';
	document.getElementById("i4").style.display='block';
}

function closeFile(){
	event.stopPropagation();
	document.getElementById("i1").style.display='none';
	document.getElementById("i2").style.display='none';
	document.getElementById("i3").style.display='none';
	document.getElementById("i4").style.display='none';
}

function sendFile(formData){
    $.ajax({
        type: "post",
        url: "/file_analyse",
        data: formData,
        dataType: "html",
        processData: false,
        contentType: false,
        async: false,
        beforeSend: function(){
            document.getElementById("i1").style.display='block';
            document.getElementById("i2").style.display='block';
            document.getElementById("i3").style.display='none';
            document.getElementById("i4").style.display='none';
        },
        success: function(res){
            document.getElementById("i1").style.display='none';
            document.getElementById("i2").style.display='none';
            document.getElementById("i3").style.display='none';
            document.getElementById("i4").style.display='none';
            $(document.body).html(res)
        },
        error: function(){
            document.getElementById("i1").style.display='block';
            document.getElementById("i2").style.display='none';
            document.getElementById("i3").style.display='block';
            document.getElementById("i4").style.display='block';
        }
    })
}

function chooseImg(){
    var imgFile = document.getElementById("formData")[0].files[0]
    var formData = new FormData();
    formData.append('file', imgFile)

    sendFile(formData)
}

function dropImg(){
    event.stopPropagation();
	event.preventDefault()
	var imgFile = event.dataTransfer.files[0]
	var formData = new FormData();
    formData.append('file', imgFile)

    sendFile(formData)
}

function sendUrl(){
    event.stopPropagation();
    var imgUrl = document.getElementById("imgUrl").value

    $.ajax({
        type: "post",
        url: "/url_analyse",
        data: JSON.stringify({"url": imgUrl}),
        dataType: "html",
        contentType: "application/json;charset=UTF-8",
        async: false,
        beforeSend: function(){
            document.getElementById("i1").style.display='block';
            document.getElementById("i2").style.display='block';
            document.getElementById("i3").style.display='none';
            document.getElementById("i4").style.display='none';
        },
        success: function(res){
            document.getElementById("i1").style.display='none';
            document.getElementById("i2").style.display='none';
            document.getElementById("i3").style.display='none';
            document.getElementById("i4").style.display='none';
            $(document.body).html(res)
        },
        error: function(){
            document.getElementById("i1").style.display='block';
            document.getElementById("i2").style.display='none';
            document.getElementById("i3").style.display='block';
            document.getElementById("i4").style.display='block';
        }
    })
}