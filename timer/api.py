from .models import Timer
from rest_framework import serializers, viewsets

class TimerSerializer(serializers.ModelSerializer):

    class Meta:
        model = Timer
        fields = '__all__'

class TimerViewSet(viewsets.ModelViewSet):
    queryset = Timer.objects.all()
    serializer_class = TimerSerializer
