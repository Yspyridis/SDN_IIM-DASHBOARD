from rest_framework import serializers 
from islanding.models import IslandingScheme
 
 
class IslandingSchemeSerializer(serializers.ModelSerializer):
 
    class Meta:
        model = IslandingScheme
        fields = (
                    'id',
                    'method_name',
                    'island_imbalance',
                    'total_imbalance',
                    'island_imbalance_after_cut',
                    'total_imbalance_after_cut',
                    'lines_to_cut',
                    'date',
                    'grid'
                )